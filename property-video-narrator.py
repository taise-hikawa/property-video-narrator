import os
import cv2
import base64
import subprocess
from openai import OpenAI

def process_property_video(video_path, seconds_per_frame=2):
    """
    音声なしの物件動画を処理し、フレーム抽出を行う

    Args:
        video_path: 処理する動画のパス
        seconds_per_frame: フレーム抽出の間隔（秒）

    Returns:
        base64エンコードされたフレームのリスト、動画の長さ（秒）
    """
    base64Frames = []

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

        # フレームをJPGに変換しbase64エンコード
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip

    video.release()

    return base64Frames, video_duration

def generate_property_summary(base64Frames, video_duration, api_key=None):
    """
    物件動画のフレームと動画の長さから物件の紹介文を生成する

    Args:
        base64Frames: base64エンコードされたフレームのリスト
        video_duration: 動画の長さ（秒）
        api_key: OpenAI APIキー（任意）

    Returns:
        物件紹介のテキスト
    """
    client = OpenAI(api_key=api_key)

    # 1秒あたり7文字で計算
    target_length = int(video_duration * 7)

    # GPT-4oで物件紹介を生成
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"""あなたは不動産専門家です。提供された物件動画のフレームに基づいて、
物件の紹介文を作成してください。この紹介文はSNSで集客するために使われます。

【重要】この紹介文は{video_duration}秒の動画に合わせたナレーション用のスクリプトです。
約{target_length}文字程度で、動画のシーンの流れに沿った内容にしてください。
動画のシーン展開を想像し、「まずこの部屋が映り、次に庭が映り...」といった
順序で紹介文を構成してください。

画像から判断できる情報のみを含め、確実に分からない情報については言及しないでください。
例えば、間取りや価格が画像から明確でない場合は推測せず、省略してください。

紹介文に含められる情報：
1. 物件タイプ（マンション、一戸建て、土地などの明らかな特徴）
2. 内装・外観の特徴（明確に見える特徴のみ）
3. 環境や立地の特徴（画像から確認できる周辺環境）
4. 視覚的なアピールポイント

紹介文は魅力的で興味を引くものにし、最後に「詳細はプロフィールのリンクから」などの
行動喚起フレーズを入れてください。"""
            },
            {
                "role": "user",
                "content": [
                    f"これは物件動画から抽出された{len(base64Frames)}枚のフレームです。これらの画像を時系列に沿って分析し、動画の流れに合ったナレーションスクリプトを作成してください。動画は{video_duration}秒あり、その長さに合わせた文章にしてください。画像から確実に判断できる情報のみを使い、推測は避けてください。",
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
        temperature=0.5,  # 創造性とバランスをとる
    )

    return response.choices[0].message.content

def generate_voiceover(text, output_path, api_key=None, voice="alloy"):
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
        instructions="SNSで物件について紹介します。明るく話して。",
        input=text,
    )

    response.stream_to_file(output_path)
    return output_path

def combine_video_audio_ffmpeg(video_path, audio_path, output_path):
    """
    FFmpegを使用して動画と音声を合成する

    Args:
        video_path: 元の動画ファイルのパス
        audio_path: 合成する音声ファイルのパス
        output_path: 出力する動画ファイルのパス

    Returns:
        合成に成功したかどうか（Boolean）
    """
    try:
        # FFmpegコマンドを構築
        command = [
            'ffmpeg',
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-y',  # 既存ファイルを上書き
            output_path
        ]

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
        return True
    except Exception as e:
        print(f"動画合成中にエラーが発生しました: {str(e)}")
        return False

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

def create_property_video(video_path, seconds_per_frame=2, api_key=None, voice="alloy", output_path=None):
    """
    物件動画から紹介文を生成し、ナレーション付き動画を作成する

    Args:
        video_path: 処理する動画のパス
        seconds_per_frame: フレーム抽出の間隔（秒）
        api_key: OpenAI APIキー（任意）
        voice: 使用する音声タイプ
        output_path: 出力する動画ファイルのパス（省略時は自動生成）

    Returns:
        物件紹介のテキストと生成された動画のパス（成功した場合）、またはエラーメッセージ
    """
    # FFmpegがインストールされているか確認
    if not check_ffmpeg_installed():
        print("FFmpegがインストールされていません。以下のコマンドでインストールしてください：")
        print("brew install ffmpeg")
        return None, "FFmpegがインストールされていません"

    print(f"動画の処理を開始: {video_path}")
    base64Frames, video_duration = process_property_video(video_path, seconds_per_frame)
    print(f"動画からフレーム {len(base64Frames)}枚を抽出しました")
    print(f"動画の長さ: {video_duration:.1f}秒")

    # 物件紹介文を生成（動画の長さを考慮）
    property_summary = generate_property_summary(base64Frames, video_duration, api_key)
    print("物件紹介文の生成が完了しました")

    # 紹介文をテキストファイルに保存
    base_path, ext = os.path.splitext(video_path)
    text_file = f"{base_path}_summary.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(property_summary)
    print(f"紹介文をテキストファイルに保存しました: {text_file}")

    # TTS で音声を生成
    audio_path = f"{base_path}_narration.mp3"
    generate_voiceover(property_summary, audio_path, api_key, voice)
    print(f"ナレーション音声を生成しました: {audio_path}")

    # 動画と音声を合成
    if output_path is None:
        output_path = f"{base_path}_with_narration{ext}"

    print(f"動画と音声を合成しています...")
    success = combine_video_audio_ffmpeg(video_path, audio_path, output_path)

    if success:
        print(f"ナレーション付き動画を作成しました: {output_path}")
        return property_summary, output_path
    else:
        return property_summary, "動画の合成に失敗しました"

# 使用例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='物件動画からナレーション付き動画を生成')
    parser.add_argument('video_path', help='処理する動画ファイルのパス')
    parser.add_argument('--interval', type=int, default=2, help='フレーム抽出間隔（秒）')
    parser.add_argument('--api-key', help='OpenAI APIキー')
    parser.add_argument('--voice', default='alloy', choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'], 
                        help='ナレーションの音声タイプ')
    parser.add_argument('--output', help='出力する動画ファイルのパス')

    args = parser.parse_args()

    summary, output_video = create_property_video(
        args.video_path,
        args.interval,
        args.api_key,
        args.voice,
        args.output
    )

    print("\n--- 物件紹介 ---\n")
    print(summary)

    if isinstance(output_video, str) and output_video.startswith("動画の合成に失敗"):
        print(f"\nエラー: {output_video}")
    else:
        print(f"\n最終動画: {output_video}")