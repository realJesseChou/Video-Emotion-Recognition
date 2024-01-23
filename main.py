import os
import time
from keras.models import load_model
from sklearn.metrics import  mutual_info_score
# 音视频分离及视频抽帧
from utils import prepare_video, clear_file, video2data, get_video_duration, generate_random_integers, get_img_emo_index
# 音频情绪识别部分
from audioR.model import audio_predict_and_sst_extrac

# 所有模型参数
from params import Params

param = Params()
model = load_model(os.path.join(param.img_model_path, "model_v6_23.hdf5"))

here = os.path.abspath(__file__)
father = os.path.dirname

sst_emo_map = {
    "积极": 0,
    "中性": 1,
    "消极": 2
}

def main(clip=5):
    start_time = time.time()
    with open(param.log_file, 'a', encoding="utf-8") as f:
        for video_name in os.listdir(param.base_path):
            clear_file(param)
            src_video_full_path = os.path.join(param.base_path, video_name)
            tar_video_full_path = os.path.join(param.mp4_path)
            # 先移动目标视频
            prepare_video(src_video_full_path, tar_video_full_path)

            print(f"正在对视频{tar_video_full_path}进行处理....")
            print(f"音视频分离及视频抽帧...")
            dirname = os.path.join(father(here), r"Repo", r"video", r"mp4")
            for name in os.listdir(dirname):
                video_path = os.path.join(dirname, name)
                duration = get_video_duration(video_path)
                intervals = generate_random_integers(duration, (int)(duration / clip))  # 每5秒分割一次。
                video2data(path=video_path,
                           cuts=intervals,
                           p=2)

            # 图像情绪识别
            print(f"图像情绪识别...")
            img_emo_index = get_img_emo_index()
            # 图像合并gif
            # print("进行图像gif合成...")
            # img2gif(param)

            # 语音情绪识别和字幕情绪识别
            print(f"语音情绪识别和字幕情绪识别...")
            audio_emo_res, sst_emo_res = audio_predict_and_sst_extrac()

            # 清理视频\图片\音频文件
            # clear_file(param)
            # 向量转化
            audio_emo_index = []
            sst_emo_index = []
            for emo1, emo2 in zip(audio_emo_res, sst_emo_res):
                index1 = sst_emo_map.get(emo1)
                index2 = sst_emo_map.get(emo2)
                if index1 is not None:
                    audio_emo_index.append(index1)
                if index2 is not None:
                    sst_emo_index.append(index2)

            # 当前视频的情感序列
            print("===" * 50)
            print(f"图像情感识别结果为：\t{img_emo_index}")
            print(f"音频情感识别结果为：\t{audio_emo_index}")
            print(f"字幕情感识别结果为：\t{sst_emo_index}")

            # 调用互信息两两计算
            # mi_bt_img_audio = adjusted_mutual_info_score(img_emo_index, audio_emo_index)
            # mi_bt_img_sst = adjusted_mutual_info_score(img_emo_index, sst_emo_index)
            # mi_bt_sst_audio = adjusted_mutual_info_score(sst_emo_index, audio_emo_index)

            mi_bt_img_audio = mutual_info_score(img_emo_index, audio_emo_index)
            mi_bt_img_sst = mutual_info_score(img_emo_index, sst_emo_index)
            mi_bt_sst_audio = mutual_info_score(sst_emo_index, audio_emo_index)
            # img_emo_index = "a"
            # audio_emo_index = "b"
            # sst_emo_index = "c"
            # mi_bt_img_audio = 1
            # mi_bt_img_sst = 2
            # mi_bt_sst_audio = 3



            # 输出
            print("===" * 50)
            print(f"当前视频中图像和音频的情感互信息值为：\t{mi_bt_img_audio}")
            print(f"当前视频中图像和字幕的情感互信息值为：\t{mi_bt_img_sst}")
            print(f"当前视频中字幕和音频的情感互信息值为：\t{mi_bt_sst_audio}")

            # log
            f.write(f"当前视频{video_name},每{clip}s采样一次进行识别,结果如下:\n")
            f.write(f"图像情感识别结果为：\t{img_emo_index}\n")
            f.write(f"音频情感识别结果为：\t{audio_emo_index}\n")
            f.write(f"字幕情感识别结果为：\t{sst_emo_index}\n")
            f.write(f"当前视频中图像和音频的情感互信息值为：\t{mi_bt_img_audio}\n")
            f.write(f"当前视频中图像和字幕的情感互信息值为：\t{mi_bt_img_sst}\n")
            f.write(f"当前视频中字幕和音频的情感互信息值为：\t{mi_bt_sst_audio}\n")
            f.write(f"处理耗时{time.time() - start_time}s\n")
            f.write("\n")
            start_time = time.time()
            f.flush()
    f.close()

if __name__ == "__main__":
    clips = [5, 7, 9]
    for clip in clips:
        main(clip=clip)