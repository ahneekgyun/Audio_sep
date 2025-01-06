import json
import random
import torch
import torchaudio
from torch.utils.data import Dataset

# Dataset을 상속받아 AudioTextDatset이라는 데이터셋을 정의함
# Dataset클래슨는 __len__(self), __getitem__(self,idx)를 필수로 구현해야함 
# 텍스트와 오디오를 포함한 딕셔너리를 리턴함

class AudioTextDataset(Dataset):
    """Can sample data from audio-text databases
    Params:
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audio clip to be sampled
    """
    def __init__(
        self,
        datafiles=[''], 
        sampling_rate=32000, 
        max_clip_len=5,
    ):
        all_data_json = []
        for datafile in datafiles:
            with open(datafile, 'r') as fp:
                data_json = json.load(fp)['data']
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json

        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate

    def __len__(self):
        return len(self.all_data_json)

    # waveform은 오디오 데이터 텐서임
    # 16000 길이로 맞추는 함수, 넘으면 자르고 부족하면 패딩
    def _cut_or_randomcrop(self, waveform):
        # waveform: [1, samples]
        # random crop
        # waveform.size(1)은 두 번째 차원의 크기
        if waveform.size(1) > self.max_length:
            random_idx = random.randint(0, waveform.size(1)-self.max_length)
            waveform = waveform[:, random_idx:random_idx+self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform

    def _read_audio(self, index):
        try:
            audio_path = self.all_data_json[index]['wav']
            # torchaudio.load는 오디오텐서, 샘플링 레이트를 반환
            audio_data, audio_rate = torchaudio.load(audio_path, channels_first=True)

            # caption이 여러개라면 첫 번째를 반환
            if isinstance(self.all_data_json[index]['caption'], list):
                text = self.all_data_json[index]['caption'][0]
            else:
                text = self.all_data_json[index]['caption']

            # drop short utterance 오디오 길이 0.5초 미만이면 버림
            if audio_data.size(1) < self.sampling_rate * 0.5:
                raise Exception(f'{audio_path} is too short, drop it ...') 
            
            return text, audio_data, audio_rate
        
        except Exception as e:
            print(f'error: {e} occurs, when loading {audio_path}')
            random_index = random.randint(0, len(self.all_data_json)-1)
            return self._read_audio(index=random_index)

    def __getitem__(self, index):
        # create a audio tensor  
        text, audio_data, audio_rate = self._read_audio(index)
        audio_len = audio_data.shape[1] / audio_rate
        # convert stero to single channel
        if audio_data.shape[0] > 1:
            # audio_data: [samples]
            audio_data = (audio_data[0] + audio_data[1]) / 2
        else:
            audio_data = audio_data.squeeze(0)
        
        # resample audio clip
        # 미리 정의한 smapling_rate 320000이 아니면 맞춰준다
        if audio_rate != self.sampling_rate:
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate, new_freq=self.sampling_rate)
        
        audio_data = audio_data.unsqueeze(0)
        
        audio_data = self._cut_or_randomcrop(audio_data)            

        data_dict = {
            'text': text, 
            'waveform': audio_data,  
            'modality': 'audio_text'
        }

        return data_dict
