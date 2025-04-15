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

def generate_timestamped_narration(base64Frames, timestamps, video_duration, api_key=None):
    """
    物件動画のフレームと動画の長さからタイムスタンプ付きの物件紹介ナレーションを生成する

    Args:
        base64Frames: base64エンコードされたフレームのリスト
        timestamps: 各フレームの時間（秒）
        video_duration: 動画の長さ（秒）
        api_key: OpenAI APIキー（任意）

    Returns:
        PropertyNarrationオブジェクト
    """
    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={
            "type": "json_schema",
            "json_schema": PROPERTY_NARRATION_SCHEMA
        },
        messages=[
            {
                "role": "system",
                "content": f"""あなたは不動産専門家です。提供された物件動画のフレームに基づいて、
物件の紹介ナレーションをタイムスタンプ付きで作成してください。

【重要】この出力は{video_duration:.1f}秒の動画に合わせたナレーション用のスクリプトです。
動画のシーン展開（各フレームのタイムスタンプ）を考慮して、適切なタイミングでナレーションが入るように
設計してください。

画像から判断できる情報のみを含め、確実に分からない情報については言及しないでください。
例えば、間取りや価格が画像から明確でない場合は推測せず、省略してください。

各セグメントは、前のセグメントの終了時間から始まり、セグメント同士が重ならないようにしてください。
最後のセグメントは動画の終了時間（{video_duration:.1f}秒）までにすべて収まるようにしてください。
"""
            },
            {
                "role": "user",
                "content": [
                    f"これは物件動画から抽出された{len(base64Frames)}枚のフレームです。これらの画像を時系列に沿って分析し、タイムスタンプ付きのナレーションスクリプトを作成してください。各フレームのタイムスタンプは以下の通りです: {', '.join([f'{t:.1f}秒' for t in timestamps])}。",
                    *map(
                        lambda x: {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{x}",
                                "detail": "low",
                            },
                        },
                        base64Frames,
                    ),
                ],
            },
        ],
        temperature=0.5,
    )

    return json.loads(response.choices[0].message.content)

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
    # 最初に無音ファイルをベースにして、各セグメントを適切な時間に挿入
    filter_complex = ""
    input_count = 1  # 最初の入力は無音ファイル
    
    # 各セグメントの入力を追加
    inputs = ['-i', silent_audio]
    for i, segment in enumerate(segment_audios):
        inputs.extend(['-i', segment['path']])
        # adelayフィルターを使用して、セグメントを適切な時間に配置
        # adelayは音声を特定の時間（ミリ秒）遅らせる
        filter_complex += f"[{input_count}]adelay={int(segment['start']*1000)}|{int(segment['start']*1000)}[a{i}];"
        input_count += 1
    
    # すべてのセグメントを無音ファイルとミックス
    # 最初に無音ファイルのラベルを設定
    filter_complex += "[0]"
    
    # 各セグメントのラベルを追加
    for i in range(len(segment_audios)):
        filter_complex += f"[a{i}]"
    
    # amixフィルターで全てをミックス（最初の無音ファイル + 全セグメント）
    filter_complex += f"amix=inputs={input_count}:duration=longest[aout]"
    
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
    narration = generate_timestamped_narration(base64Frames, timestamps, video_duration, api_key)
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