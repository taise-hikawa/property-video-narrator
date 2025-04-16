import os
import cv2
import base64
import subprocess
import json
from openai import OpenAI

# Pydanticモデルの代わりにJSONスキーマを定義
PROPERTY_NARRATION_SCHEMA = {
    "name": "property_narration",
    "description": "物件紹介のナレーション構造",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "description": "タイムスタンプ付きのナレーションセグメント",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "number",
                            "description": "セグメント開始時間（秒）"
                        },
                        "end_time": {
                            "type": "number",
                            "description": "セグメント終了時間（秒）"
                        },
                        "text": {
                            "type": "string",
                            "description": "このセグメントのナレーションテキスト"
                        }
                    },
                    "required": ["start_time", "end_time", "text"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["segments"],
        "additionalProperties": False
    }
}

PROPERTY_FEATURES_SCHEMA = {
    "name": "property_room_features",
    "description": "物件の部屋の特徴構造",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_time": {"type": "number"},
                        "end_time": {"type": "number"},
                        "features": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "このセグメントで見える特徴のリスト"
                        }
                    },
                    "required": ["start_time", "end_time", "features"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["segments"],
        "additionalProperties": False
    }
}

def process_property_video(video_path, seconds_per_frame=1.5):
    """
    音声なしの物件動画を処理し、フレーム抽出を行う

    Args:
        video_path: 処理する動画のパス
        seconds_per_frame: フレーム抽出の間隔（秒）

    Returns:
        base64エンコードされたフレームのリスト、動画の長さ（秒）
    """
    base64Frames = []
    timestamps = []

    # 動画からフレームを抽出
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)
    video_duration = total_frames / fps

    curr_frame = 0
    while curr_frame < total_frames - 1:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break

        # 現在のタイムスタンプを記録
        timestamp = curr_frame / fps
        timestamps.append(timestamp)

        # フレームをJPGに変換しbase64エンコード
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip

    video.release()

    return base64Frames, timestamps, video_duration

def add_length_limits_to_features(features):
    for segment in features['segments']:
        # セグメントの長さを計算（秒）
        segment_duration = segment['end_time'] - segment['start_time']
        # 1秒あたり5文字で最大文字数を計算
        max_chars = int(segment_duration * 5)
        # 文字数制限情報を追加
        segment['max_chars'] = max_chars

    return features

def generate_timestamped_narration(base64Frames, timestamps, video_duration, seconds_per_frame, api_key=None):
    """
    物件動画のフレームと動画の長さからタイムスタンプ付きの物件紹介ナレーションを生成する

    Args:
        base64Frames: base64エンコードされたフレームのリスト
        timestamps: 各フレームの時間（秒）
        video_duration: 動画の長さ（秒）
        seconds_per_frame: フレーム抽出間隔（秒）
        api_key: OpenAI APIキー（任意）

    Returns:
        PropertyNarrationオブジェクト
    """
    client = OpenAI(api_key=api_key)

    # Step 1: 各セグメントの特徴を抽出
    features_response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_schema", "json_schema": PROPERTY_FEATURES_SCHEMA},
        messages=[
            {
                "role": "system",
                "content": f"""不動産専門家として、各シーンで見える特徴を箇条書きで列挙してください。
各特徴は簡潔な名詞句で記述してください。

【最重要】同じ部屋や同じシーンは必ず1つのセグメントにまとめてください。例えば：
- 外観の映像が複数枚続く場合 → 1つの「外観」セグメント
- 玄関の映像が複数枚続く場合 → 1つの「玄関」セグメント

動画内の主要なシーンを特定し、それぞれを独立したセグメントとして扱ってください。
少なくとも以下のようなシーンを区別してください：
- 建物外観
- 玄関・エントランス
- リビング・ダイニング
- キッチン
- 寝室
- バスルーム・トイレ
- バルコニー・外部視点

各フレームは以下のタイムスタンプで撮影されています：
{', '.join([f'{t:.1f}秒' for t in timestamps])}

セ【セグメント時間の正確な指定方法】
- 各フレームの抽出時間：フレーム0が0.0秒、フレーム1が{seconds_per_frame}秒、フレーム2が{2*seconds_per_frame}秒...
- セグメントの開始時間：そのシーンの最初のフレームの時間
- セグメントの終了時間：そのシーンの最後のフレームの時間（次のフレームはもう別のシーン）

例（{seconds_per_frame}秒間隔でフレームを抽出した場合）：
- フレーム0-3がキッチン、フレーム4から寝室 → [0.0秒-3.0秒]がキッチンセグメント
- フレーム4-7が寝室、フレーム8からバスルーム → [4.0秒-7.0秒]が寝室セグメント
"""
            },
            {
                "role": "user",
                "content": [
                    f"これは物件動画から{seconds_per_frame}秒ごとに抽出された{len(base64Frames)}枚のフレームです。各フレームの特徴を時系列に沿って分析してください。",
                    *[{
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{frame}",
                            "detail": "high"
                        }
                    } for frame in base64Frames]
                ]
            }
        ],
        temperature=0.5
    )

    print("\n特徴:")
    features = json.loads(features_response.choices[0].message.content)
    for i, segment in enumerate(features['segments']):
        print(f"\nセグメント {i+1} [{segment['start_time']:.1f}秒-{segment['end_time']:.1f}秒]:")
        for feature in segment['features']:
            print(f"  - {feature}")
    print("\n")
    features_with_limits = add_length_limits_to_features(features)
    # Step 2: 特徴からナレーションを生成
    narration_response = client.chat.completions.create(
        model="gpt-4.1",
        response_format={"type": "json_schema", "json_schema": PROPERTY_NARRATION_SCHEMA},
        messages=[
            {
                "role": "system",
                "content": f"""
あなたはSNSで不動産を紹介するインフルエンサーです。提供された特徴リストから簡潔なナレーションを作成してください。

【最重要】文字数制限を厳守してください：
- 1秒あたり最大5文字まで。カウントする際は読み仮名でカウントして。
- 各セグメントの文字数上限 = セグメントの秒数 × 5

例：
- 6秒セグメント → 最大30文字
- 9秒セグメント → 最大45文字

必ず文字数をカウントし、制限を超えないようにしてください。内容よりも文字数制限を優先してください。

セグメント間の繋がりを意識し、一つの流れのある紹介になるよう心がけてください。
「さて」「こちらが」「そして」「続いて」などの接続表現を使いながら、
家の中を案内しているような自然な流れを作ってください。

親しみやすいSNS風の簡潔な話し言葉を使用してください。
各セグメントを作成する前に、必ず最大文字数を計算し、その範囲に収めてください。
最初のセグメントは「を紹介します」という文言が入ると良い。
                """
            },
            {
                "role": "user",
                "content": f"以下の特徴リストから物件紹介のナレーションを作成してください。各セグメントの「max_chars」に示された文字数以内に必ず収めてください:\n{json.dumps(features_with_limits, ensure_ascii=False, indent=2)}"
            }
        ],
        temperature=0.8
    )

    return json.loads(narration_response.choices[0].message.content)

def generate_segment_voiceover(text, output_path, api_key=None, voice="alloy"):
    """
    テキストから音声ナレーションを生成する

    Args:
        text: 読み上げるテキスト
        output_path: 出力する音声ファイルのパス
        api_key: OpenAI APIキー（任意）
        voice: 使用する音声タイプ（デフォルト：alloy）

    Returns:
        生成された音声ファイルのパス
    """
    client = OpenAI(api_key=api_key)

    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        instructions="SNSで物件について紹介します。明るくゆっくり話して。",
        input=text,
    )

    response.stream_to_file(output_path)
    return output_path

def combine_video_with_segmented_audio(video_path, narration, temp_dir, output_path, api_key=None, voice="alloy"):
    """
    動画とセグメント分けされたナレーションを組み合わせる

    Args:
        video_path: 元の動画ファイルのパス
        narration: PropertyNarrationオブジェクト
        temp_dir: 一時ファイル保存ディレクトリ
        output_path: 出力する動画ファイルのパス
        api_key: OpenAI APIキー（任意）
        voice: 使用する音声タイプ

    Returns:
        合成に成功したかどうか（Boolean）
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    # 無音の音声ファイルを作成
    silent_audio = os.path.join(temp_dir, "silent.mp3")
    subprocess.run([
        'ffmpeg',
        '-f', 'lavfi',
        '-i', 'anullsrc=r=44100:cl=stereo',
        '-t', str(narration['segments'][-1]['end_time']),
        '-q:a', '9',
        '-c:a', 'libmp3lame',
        '-y',
        silent_audio
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 各セグメントの音声を生成
    segment_audios = []
    for i, segment in enumerate(narration['segments']):
        segment_audio = os.path.join(temp_dir, f"segment_{i}.mp3")
        generate_segment_voiceover(segment['text'], segment_audio, api_key, voice)
        segment_audios.append({
            'path': segment_audio,
            'start': segment['start_time'],
            'end': segment['end_time']
        })
    
    # FFmpegコマンドを構築
    filter_complex = ""
    input_count = 1  # 最初の入力は無音ファイル
    
    # 各セグメントの入力を追加
    inputs = ['-i', silent_audio]
    for i, segment in enumerate(segment_audios):
        inputs.extend(['-i', segment['path']])
        # adelayフィルターを使用して、セグメントを適切な時間に配置
        filter_complex += f"[{input_count}]adelay={int(segment['start']*1000)}|{int(segment['start']*1000)}[a{i}];"
        input_count += 1
    
    # すべてのストリームをミックス
    if len(segment_audios) > 0:
        # 無音ファイルと全セグメントのミックス
        filter_complex += "[0]"  # 無音ファイル
        for i in range(len(segment_audios)):
            filter_complex += f"[a{i}]"  # 各セグメント
        filter_complex += f"amix=inputs={input_count}:duration=longest[aout]"
    else:
        # セグメントがない場合は無音ファイルをそのまま使用
        filter_complex += "[0]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[aout]"
    
    # 最終的なFFmpegコマンド
    command = [
        'ffmpeg',
        *inputs,
        '-i', video_path,
        '-filter_complex', filter_complex,
        '-map', f"{input_count}:v:0",  # 動画のビデオストリーム
        '-map', '[aout]',  # ミックスした音声
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        '-y',
        output_path
    ]
    
    print(f"実行するFFmpegコマンド: {' '.join(command)}")
    
    # FFmpegを実行
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # エラーチェック
    if result.returncode != 0:
        print(f"FFmpeg実行中にエラーが発生しました:\n{result.stderr}")
        return False
    
    # 一時ファイルを削除
    for segment in segment_audios:
        os.remove(segment['path'])
    os.remove(silent_audio)
    
    return True

def check_ffmpeg_installed():
    """
    FFmpegがインストールされているか確認する

    Returns:
        インストールされているかどうか（Boolean）
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def create_property_video_with_segments(video_path, seconds_per_frame=1.5, api_key=None, voice="alloy", output_path=None):
    """
    物件動画からタイムスタンプ付きナレーションを生成し、ナレーション付き動画を作成する

    Args:
        video_path: 処理する動画のパス
        seconds_per_frame: フレーム抽出の間隔（秒）
        api_key: OpenAI APIキー（任意）
        voice: 使用する音声タイプ
        output_path: 出力する動画ファイルのパス（省略時は自動生成）

    Returns:
        物件紹介のナレーションオブジェクトと生成された動画のパス（成功した場合）、またはエラーメッセージ
    """
    # FFmpegがインストールされているか確認
    if not check_ffmpeg_installed():
        print("FFmpegがインストールされていません。以下のコマンドでインストールしてください：")
        print("brew install ffmpeg")
        return None, "FFmpegがインストールされていません"

    print(f"動画の処理を開始: {video_path}")
    base64Frames, timestamps, video_duration = process_property_video(video_path, seconds_per_frame)
    print(f"動画からフレーム {len(base64Frames)}枚を抽出しました")
    print(f"動画の長さ: {video_duration:.1f}秒")

    # タイムスタンプ付き物件紹介ナレーションを生成
    narration = generate_timestamped_narration(
        base64Frames,
        timestamps,
        video_duration,
        seconds_per_frame,  # この引数を追加
        api_key
    )
    print("タイムスタンプ付き物件紹介ナレーションの生成が完了しました")

    # ナレーションをJSON形式でファイルに保存
    base_path, ext = os.path.splitext(video_path)
    json_file = f"{base_path}_narration.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(narration, f, indent=2, ensure_ascii=False)
    print(f"ナレーションをJSONファイルに保存しました: {json_file}")
    print(narration)
    # フルテキストをテキストファイルとしても保存
    text_file = f"{base_path}_summary.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        for segment in narration['segments']:
            f.write(f"[{segment['start_time']:.1f}秒-{segment['end_time']:.1f}秒] {segment['text']}\n")
    print(f"ナレーションをテキストファイルに保存しました: {text_file}")

    # 動画と音声を合成
    if output_path is None:
        output_path = f"{base_path}_with_narration{ext}"

    # 一時ファイル用ディレクトリ
    temp_dir = f"{base_path}_temp"

    print(f"動画とセグメント化されたナレーションを合成しています...")
    success = combine_video_with_segmented_audio(
        video_path, narration, temp_dir, output_path, api_key, voice
    )

    if success:
        print(f"タイムスタンプ付きナレーション動画を作成しました: {output_path}")
        return narration, output_path
    else:
        return narration, "動画の合成に失敗しました"

# 使用例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='物件動画からタイムスタンプ付きナレーション動画を生成')
    parser.add_argument('video_path', help='処理する動画ファイルのパス')
    parser.add_argument('--interval', type=float, default=1.5, help='フレーム抽出間隔（秒）')
    parser.add_argument('--api-key', help='OpenAI APIキー')
    parser.add_argument('--voice', default='alloy', choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
                        help='ナレーションの音声タイプ')
    parser.add_argument('--output', help='出力する動画ファイルのパス')

    args = parser.parse_args()

    narration, output_video = create_property_video_with_segments(
        args.video_path,
        args.interval,
        args.api_key,
        args.voice,
        args.output
    )

    print("\n--- 物件紹介ナレーション ---\n")
    print("\nセグメント:")
    for i, segment in enumerate(narration['segments']):
        print(f"  [{segment['start_time']:.1f}秒-{segment['end_time']:.1f}秒] {segment['text']}")

    if isinstance(output_video, str) and output_video.startswith("動画の合成に失敗"):
        print(f"\nエラー: {output_video}")
    else:
        print(f"\n最終動画: {output_video}")