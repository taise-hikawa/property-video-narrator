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

def add_length_limits_to_features(features, language='ja'):
    """
    Add character limits based on language
    ja: 4 chars/sec (Japanese)
    en: 8 chars/sec (English)
    zh: 5 chars/sec (Chinese)
    """
    chars_per_second = {
        'ja': 4,
        'en': 7,
        'zh': 3
    }

    rate = chars_per_second.get(language, 4)

    for segment in features['segments']:
        segment_duration = segment['end_time'] - segment['start_time']
        max_chars = int(segment_duration * rate)
        segment['max_chars'] = max_chars

    return features

def generate_timestamped_narration(base64Frames, timestamps, video_duration, seconds_per_frame, property_info=None, api_key=None, language='ja'):
    """
    物件動画のフレームと動画の長さからタイムスタンプ付きの物件紹介ナレーションを生成する

    Args:
        base64Frames: base64エンコードされたフレームのリスト
        timestamps: 各フレームの時間（秒）
        video_duration: 動画の長さ（秒）
        seconds_per_frame: フレーム抽出間隔（秒）
        property_info: 物件情報（文字列）
        api_key: OpenAI APIキー（任意）

    Returns:
        PropertyNarrationオブジェクト
    """
    client = OpenAI(api_key=api_key)

    # 物件情報があれば、それを含むプロンプトを各言語用に作成
    property_info_prompts = {
        'en': f"""
[IMPORTANT PROPERTY INFORMATION]
Please naturally incorporate the following information into the narration:
{property_info}
""",
        'zh': f"""
【重要房产信息】
请将以下信息自然地融入解说中：
{property_info}
""",
        'ja': f"""
【重要な物件情報】
以下の情報を必ずナレーションの中に自然な形で組み込んでください：
{property_info}
"""
    }

    property_info_prompt = property_info_prompts.get(language, "") if property_info else ""

    # Step 1: 各セグメントの特徴を抽出
    features_response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_schema", "json_schema": PROPERTY_FEATURES_SCHEMA},
        messages=[
            {
                "role": "system",
                "content": f"""
                不動産専門家として、各シーンで見える特徴を箇条書きで列挙してください。
各特徴は簡潔な名詞句で記述してください。

【最重要】動画の正確なセグメント分割
- フレーム0が0.0秒、フレーム1が{seconds_per_frame}秒、フレーム2が{2*seconds_per_frame}秒...と認識してください
- 必ず最初のセグメントは0.0秒から始めてください
- 最後のセグメントは必ず{video_duration}秒で終わるようにしてください
- セグメント間にはギャップを作らないでください（前のセグメントの終了時間 = 次のセグメントの開始時間）

【空間認識ルール】
- 同じ部屋や同じシーンは必ず1つのセグメントにまとめてください
- 以下の空間タイプを区別して判断してください：
  - 建物外観
  - 玄関・エントランス
  - リビング・ダイニング
  - キッチン
  - 寝室・ベッドルーム
  - 洗面所・パウダールーム
  - 浴室・バスルーム
  - ベランダ・バルコニー

【セグメント時間の正確な指定方法】
1. 最初のセグメントは0.0秒から開始
2. 空間/シーンが切り替わる場所でセグメントを区切る
3. 各セグメントの終了時間 = 次のセグメントの開始時間
4. 最後のセグメントの終了時間 = {video_duration}秒

例：
- フレーム0-3が外観、フレーム4-7が玄関、フレーム8からリビング →
  [0.0秒-{3*seconds_per_frame:.1f}秒]が外観セグメント
  [{4*seconds_per_frame:.1f}秒-{7*seconds_per_frame:.1f}秒]が玄関セグメント
  [{8*seconds_per_frame:.1f}秒-{video_duration:.1f}秒]がリビングセグメント

フレームの先に映像がある場合も、最後のフレームのシーンは動画の終了時間まで続くと判断してください。"""
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

    print("\n特徴:")
    # 文字数制限を追加
    features_with_limits = add_length_limits_to_features(features, language)
    
    # 全セグメント数を追加
    total_segments = len(features_with_limits['segments'])
    for i, segment in enumerate(features_with_limits['segments']):
        segment['segment_index'] = i
        segment['total_segments'] = total_segments
    
    # Step 2: セグメントごとにナレーションを生成
    narration = {
        "segments": []
    }
    first_segment = features_with_limits['segments'][0]
    last_segment = features_with_limits['segments'][-1]

    first_duration = first_segment['end_time'] - first_segment['start_time']
    last_duration = last_segment['end_time'] - last_segment['start_time']

    # 最初と最後のセグメントのうち、長い方のインデックスを選択
    longest_segment_index = 0 if first_duration >= last_duration else len(features_with_limits['segments']) - 1

    for i, segment in enumerate(features_with_limits['segments']):
        print(f"セグメント {i+1}/{total_segments} のナレーションを生成中...")

        # セグメントごとにナレーションを生成
        narration_text = generate_segment_narration(client, segment, property_info, language, i == longest_segment_index)

        # 生成されたナレーションをセグメントに追加
        narration['segments'].append({
            "start_time": segment['start_time'],
            "end_time": segment['end_time'],
            "text": narration_text
        })

        print(f"  セグメント {i+1} ナレーション: {narration_text}")
        print(f"  文字数: {len(narration_text)}/{segment['max_chars']}")

    return narration

def generate_segment_narration(client, segment, property_info=None, language='ja', include_property_info=False):
    """
    セグメントごとにナレーションを生成する

    Args:
        client: OpenAI APIクライアント
        segment: ナレーションを生成するセグメント情報
        property_info: 物件情報（文字列）
        language: 言語設定（デフォルト: 日本語）

    Returns:
        生成されたナレーションテキスト
    """
    property_instruction = ""
    if property_info and include_property_info:

        if include_property_info:
            # このセグメントに物件情報を含める
            if language == 'ja':
                property_instruction = f"""
【重要な物件情報】
以下の情報をこのセグメントのナレーションに自然な形で含めてください：
{property_info}
"""
            elif language == 'en':
                property_instruction = f"""
[IMPORTANT PROPERTY INFORMATION]
Please include the following information in this segment:
{property_info}
"""
            elif language == 'zh':
                property_instruction = f"""
【重要房产信息】
请在此片段中自然地融入以下信息：
{property_info}
"""
        else:
            # このセグメントでは物件情報を含めない
            if language == 'ja':
                property_instruction = f"""
【重要な注意点】
以下の情報は別のセグメントで紹介するので、このセグメントでは含めないでください：
{property_info}
"""
            elif language == 'en':
                property_instruction = f"""
[IMPORTANT NOTE]
The following information will be introduced in another segment. Do not include it in this segment:
{property_info}
"""
            elif language == 'zh':
                property_instruction = f"""
【重要提示】
以下信息将在另一个片段中介绍。请不要在此片段中包含它：
{property_info}
"""
    # Language-specific prompts
    language_prompts = {
        'en': f"""
Please output in English.
You are a professional real estate influencer. Create engaging narration for a single segment of a property video.

{property_instruction}

[CRITICAL - MUST FOLLOW]
- Strictly adhere to the character limit specified by 'max_chars' for this segment
- Count every character, including spaces and punctuation
- Do not exceed the limit under any circumstances
- Example: "Spacious living room" = 19 characters

[Expression Guidelines]
- Use natural, SNS-friendly language that's brief and impactful
- Focus on emotional value rather than just physical descriptions
- Example: "Large windows" → "Sun-filled windows"
- Keep descriptions concise and engaging
""",
        'zh': f"""
请用中文输出。
您是一位专业的房地产博主。请根据给定的特征为房产视频的单一片段创建吸引人的解说。

{property_instruction}

【最重要 - 必须遵守】
- 严格遵守'max_chars'指定的字数限制
- 计算每个字符，包括标点符号
- 在任何情况下都不能超过限制
- 示例："宽敞明亮的客厅" = 7个字符

【表达准则】
- 使用自然、适合社交媒体的语言
- 注重情感价值而不是单纯的物理描述
- 示例："大窗户" → "阳光充沛的窗户"
- 保持描述简洁有力
""",
        'ja': f"""
日本語でアウトプットしてください。
あなたはSNSで不動産を紹介するインフルエンサーです。提供された特徴リストから物件紹介動画の1つのセグメントのナレーションを作成してください。

{property_instruction}

【最重要・厳守】バランスの取れた表現
- 文字数はmax_chars値を絶対に超えないようにしてください
- 文字数は読み方（ひらがな）で厳密にカウント
- 例: 「緑豊かな外観」→「みどりゆたかながいかん」で13文字

【表現のバランス】
- 必ず適切に漢字を使用する（全てひらがなにしない）
- 例: 「ひろびろしたリビング」ではなく「広々としたリビング」
- 簡潔さを優先しつつ、自然な日本語にする

【付加価値のある表現】
- 物理的特徴から感覚的・情緒的な価値を簡潔に伝えてください
  例: 「大きな窓」→「光あふれる窓」
  例: 「木目調の壁」→「落ち着く木目調」
- 感覚的表現は短く端的に

親しみやすいSNS風の簡潔な話し言葉を使用してください。
"""
    }

    # 特定のセグメントのプロンプト調整
    segment_position_prompt = ""
    if 'segment_index' in segment and 'total_segments' in segment:
        if segment['segment_index'] == 0:
            if language == 'ja':
                segment_position_prompt = "これは紹介の最初のセグメントです。「紹介します」という表現を含めてください。"
            elif language == 'en':
                segment_position_prompt = "This is the first segment of the introduction. Please include 'Let me show you' in your narration."
            elif language == 'zh':
                segment_position_prompt = "这是介绍的第一个片段。请在解说中包含让我为您介绍。"
        elif segment['segment_index'] == segment['total_segments'] - 1:
            if language == 'ja':
                segment_position_prompt = "これは紹介の最後のセグメントです。締めくくりの表現を使用してください。"
            elif language == 'en':
                segment_position_prompt = "This is the last segment of the introduction. Please use a closing expression."
            elif language == 'zh':
                segment_position_prompt = "这是介绍的最后一个片段。请使用结束语。"

    # セグメント固有のプロンプトを追加
    prompt = language_prompts.get(language, language_prompts['ja']) + "\n" + segment_position_prompt

    # 単一セグメントのナレーション生成
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": f"以下の特徴から、{segment['max_chars']}文字以内のナレーションを作成してください。セグメント時間: {segment['start_time']:.1f}秒から{segment['end_time']:.1f}秒\n特徴リスト: {', '.join(segment['features'])}"
            }
        ],
        temperature=0.4
    )
    
    return response.choices[0].message.content.strip()


def generate_segment_voiceover_niji_voice(text, output_path, api_key=None):
    """
    テキストから音声ナレーションを生成する (NijiVoice版)

    Args:
        text: 読み上げるテキスト
        output_path: 出力する音声ファイルのパス
        api_key: NijiVoice APIキー
        voice: ボイスアクターID (OpenAIのvoice引数と互換性を持たせるため同名のパラメータを使用)

    Returns:
        生成された音声ファイルのパス
    """
    import requests

    if not api_key:
        raise ValueError("NijiVoice APIキーが必要です")

    # voiceにはボイスアクターIDが入ります（OpenAIの関数と互換性を持たせるため）
    voice_actor_id = "69f3f0fa-860b-46ff-a7d5-6a950500fb40"

    url = f"https://api.nijivoice.com/api/platform/v1/voice-actors/{voice_actor_id}/generate-voice"

    # デフォルトのフォーマットはwavに設定
    # 速度は元々のOpenAI TTSのデフォルトに近いと思われる値を設定
    payload = {
        "script": text,
        "speed": "0.9",
        "format": "mp3"
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "x-api-key": api_key
    }

    try:
        # NijiVoice APIを呼び出す
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # エラーチェック

        # レスポンスから音声URLを取得
        audio_url = response.json()["generatedVoice"]["audioFileUrl"]

        # 音声ファイルをダウンロード
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()

        # 音声ファイルを保存
        with open(output_path, "wb") as f:
            f.write(audio_response.content)
    
        return output_path

    except requests.exceptions.RequestException as e:
        print(f"NijiVoice API呼び出し中にエラーが発生しました: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"エラーレスポンス: {e.response.text}")
        return None

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

    # Use with_streaming_response instead of stream_to_file
    with open(output_path, 'wb') as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
    return output_path

def combine_video_with_segmented_audio(video_path, narration, temp_dir, output_path, api_key=None, niji_voice_api_key=None, voice="alloy", volume_level=2.0, lang="ja"):
    """
    動画とセグメント分けされたナレーションを組み合わせる

    Args:
        video_path: 元の動画ファイルのパス
        narration: PropertyNarrationオブジェクト
        temp_dir: 一時ファイル保存ディレクトリ
        output_path: 出力する動画ファイルのパス
        api_key: OpenAI APIキー（任意）
        voice: 使用する音声タイプ
        volume_level: 音量倍率 (1.0=等倍, 2.0=2倍, etc.)
        lang: 言語

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
        if lang=="ja" and niji_voice_api_key:
            generate_segment_voiceover_niji_voice(segment['text'], segment_audio, niji_voice_api_key)
        else:
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
        filter_complex += f"amix=inputs={input_count}:duration=longest:normalize=0,dynaudnorm,volume={volume_level}[aout]"
    else:
        # セグメントがない場合は無音ファイルをそのまま使用
        filter_complex += "[0]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,volume={volume_level}[aout]"

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

# Modify create_property_video_with_segments to include language parameter
def create_property_video_with_segments(video_path, seconds_per_frame=1.5, property_info=None, api_key=None, niji_voice_api_key=None, voice="alloy", output_path=None, language='ja', volume_level=2.0):
    """
    物件動画からタイムスタンプ付きナレーションを生成し、ナレーション付き動画を作成する

    Args:
        video_path: 処理する動画のパス
        seconds_per_frame: フレーム抽出の間隔（秒）
        property_info: 物件情報（文字列）
        api_key: OpenAI APIキー（任意）
        voice: 使用する音声タイプ
        output_path: 出力する動画ファイルのパス（省略時は自動生成）
        language: ナレーション言語 ('ja': 日本語, 'en': 英語, 'zh': 中国語)
        volume_level: 音量倍率 (1.0=等倍, 2.0=2倍, etc.)

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
        seconds_per_frame,
        property_info,
        api_key,
        language
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
        video_path, narration, temp_dir, output_path, api_key, niji_voice_api_key, voice, volume_level, language
    )

    if success:
        print(f"タイムスタンプ付きナレーション動画を作成しました: {output_path}")
        return narration, output_path
    else:
        return narration, "動画の合成に失敗しました"

# 使用例
# 使用例
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='物件動画からタイムスタンプ付きナレーション動画を生成')
    parser.add_argument('video_path', help='処理する動画ファイルのパス')
    parser.add_argument('--interval', type=float, default=1.5, help='フレーム抽出間隔（秒）')
    parser.add_argument('--property-info', help='物件情報（例：駅から7分、品川、家賃13万）')
    parser.add_argument('--api-key', help='OpenAI APIキー')
    parser.add_argument('--niji-voice-api-key', help='にじボイスのAPIキー')
    parser.add_argument('--voice', default='alloy', choices=['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'],
                        help='ナレーションの音声タイプ')
    parser.add_argument('--output', help='出力する動画ファイルのパス')
    parser.add_argument('--language', default='ja', choices=['ja', 'en', 'zh'],
                        help='ナレーション言語 (ja: 日本語, en: 英語, zh: 中国語)')
    parser.add_argument('--volume', type=float, default=2.0, help='音量倍率 (1.0=等倍, 2.0=2倍など)')

    args = parser.parse_args()

    narration, output_video = create_property_video_with_segments(
        args.video_path,
        args.interval,
        args.property_info,
        args.api_key,
        args.niji_voice_api_key,
        args.voice,
        args.output,
        args.language,
        args.volume
    )

    print("\n--- 物件紹介ナレーション ---\n")
    print("\nセグメント:")
    for i, segment in enumerate(narration['segments']):
        print(f"  [{segment['start_time']:.1f}秒-{segment['end_time']:.1f}秒] {segment['text']}")

    if isinstance(output_video, str) and output_video.startswith("動画の合成に失敗"):
        print(f"\nエラー: {output_video}")
    else:
        print(f"\n最終動画: {output_video}")