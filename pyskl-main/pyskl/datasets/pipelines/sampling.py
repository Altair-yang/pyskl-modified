# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import torch

import numpy as np

from ..builder import PIPELINES
import random


@PIPELINES.register_module()
class UniformSampleFrames:
    """Uniformly sample frames from the video.

    To sample an n-frame clip from the video. UniformSampleFrames basically
    divide the video into n segments of equal length and randomly sample one
    frame from each segment. To make the testing results reproducible, a
    random seed is set during testing, to make the sampling results
    deterministic.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "clip_len", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 test_mode=False,
                 float_ok=False,
                 p_interval=1,
                 seed=255,
                 order=False  # 是否使用增加的2行排序代码
                 ):

        self.order = order
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.float_ok = float_ok
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)

        if self.float_ok:
            warnings.warn('When float_ok == True, there will be no loop.')

    def _get_train_clips(self, num_frames, clip_len):
        """Uniformly sample indices for training clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        for clip_idx in range(self.num_clips):
            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if self.float_ok:
                interval = (num_frames - 1) / clip_len
                offsets = np.arange(clip_len) * interval
                inds = np.random.rand(clip_len) * interval + offsets
                inds = inds.astype(np.float32)
            elif num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array(
                    [i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            inds = inds + off
            num_frames = old_num_frames

            allinds.append(inds)

        return np.concatenate(allinds)

    def _get_test_clips(self, num_frames, clip_len):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.float_ok:
            interval = (num_frames - 1) / clip_len
            offsets = np.arange(clip_len) * interval
            inds = np.concatenate([
                np.random.rand(clip_len) * interval + offsets
                for i in range(self.num_clips)
            ]).astype(np.float32)

        all_inds = []

        for i in range(self.num_clips):

            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                # mean_acc 0.9381
                start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)

                # mean_acc 0.8875
                # all_frames = np.array(list(range(num_frames)))
                # inds = np.zeros((clip_len,), all_frames.dtype)
                # inds[:all_frames.shape[0]] = all_frames

            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                bsize = np.diff(bids)
                bst = bids[:clip_len]
                offset = np.random.randint(bsize)
                inds = bst + offset

            all_inds.append(inds + off)
            num_frames = old_num_frames

        return np.concatenate(all_inds)

    def __call__(self, results):
        num_frames = results['total_frames']

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.clip_len)
        else:
            inds = self._get_train_clips(num_frames, self.clip_len)

        inds = np.mod(inds, num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        # 22.07.27 add
        if self.order:
            for i in range(self.num_clips):
                inds[i:i+self.clip_len].sort()

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(np.int)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds if self.float_ok else inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode}, '
                    f'seed={self.seed})')
        return repr_str


@PIPELINES.register_module()
class UniformSample(UniformSampleFrames):
    pass


@PIPELINES.register_module()
class SampleFrames:
    """Sample frames from the video.

    Required keys are "total_frames", "start_index" , added or modified keys
    are "frame_inds", "frame_interval" and "num_clips".

    Args:
        clip_len (int): Frames of each sampled output clip.
        frame_interval (int): Temporal interval of adjacent sampled frames.
            Default: 1.
        num_clips (int): Number of clips to be sampled. Default: 1.
        temporal_jitter (bool): Whether to apply temporal jittering.
            Default: False.
        twice_sample (bool): Whether to use twice sample when testing.
            If set to True, it will sample frames with and without fixed shift,
            which is commonly used for testing in TSM model. Default: False.
        out_of_bound_opt (str): The way to deal with out of bounds frame
            indexes. Available options are 'loop', 'repeat_last'.
            Default: 'loop'.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        start_index (None): This argument is deprecated and moved to dataset
            class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
            see this: https://github.com/open-mmlab/mmaction2/pull/89.
        keep_tail_frames (bool): Whether to keep tail frames when sampling.
            Default: False.
    """

    def __init__(self,
                 clip_len,
                 frame_interval=1,
                 num_clips=1,
                 temporal_jitter=False,
                 twice_sample=False,
                 out_of_bound_opt='loop',
                 test_mode=False,
                 start_index=None,
                 keep_tail_frames=False):

        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.temporal_jitter = temporal_jitter
        self.twice_sample = twice_sample
        self.out_of_bound_opt = out_of_bound_opt
        self.test_mode = test_mode
        self.keep_tail_frames = keep_tail_frames
        assert self.out_of_bound_opt in ['loop', 'repeat_last']

        if start_index is not None:
            warnings.warn('No longer support "start_index" in "SampleFrames", '
                          'it should be set in dataset class, see this pr: '
                          'https://github.com/open-mmlab/mmaction2/pull/89')

    def _get_train_clips(self, num_frames):
        """Get clip offsets in train mode.

        It will calculate the average interval for selected frames,
        and randomly shift them within offsets between [0, avg_interval].
        If the total number of frames is smaller than clips num or origin
        frames length, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in train mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval

        if self.keep_tail_frames:
            avg_interval = (num_frames - ori_clip_len + 1) / float(
                self.num_clips)
            if num_frames > ori_clip_len - 1:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = (base_offsets + np.random.uniform(
                    0, avg_interval, self.num_clips)).astype(np.int)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        else:
            avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips

            if avg_interval > 0:
                base_offsets = np.arange(self.num_clips) * avg_interval
                clip_offsets = base_offsets + np.random.randint(
                    avg_interval, size=self.num_clips)
            elif num_frames > max(self.num_clips, ori_clip_len):
                clip_offsets = np.sort(
                    np.random.randint(
                        num_frames - ori_clip_len + 1, size=self.num_clips))
            elif avg_interval == 0:
                ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
                clip_offsets = np.around(np.arange(self.num_clips) * ratio)
            else:
                clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)

        return clip_offsets

    def _get_test_clips(self, num_frames):
        """Get clip offsets in test mode.

        Calculate the average interval for selected frames, and shift them
        fixedly by avg_interval/2. If set twice_sample True, it will sample
        frames together without fixed shift. If the total number of frames is
        not enough, it will return all zero indices.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices in test mode.
        """
        ori_clip_len = self.clip_len * self.frame_interval
        avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
        if num_frames > ori_clip_len - 1:
            base_offsets = np.arange(self.num_clips) * avg_interval
            clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
            if self.twice_sample:
                clip_offsets = np.concatenate([clip_offsets, base_offsets])
        else:
            clip_offsets = np.zeros((self.num_clips, ), dtype=np.int)
        return clip_offsets

    def _sample_clips(self, num_frames):
        """Choose clip offsets for the video in a given mode.

        Args:
            num_frames (int): Total number of frame in the video.

        Returns:
            np.ndarray: Sampled frame indices.
        """
        if self.test_mode:
            clip_offsets = self._get_test_clips(num_frames)
        else:
            clip_offsets = self._get_train_clips(num_frames)

        return clip_offsets

    def __call__(self, results):
        """Perform the SampleFrames loading.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        total_frames = results['total_frames']

        clip_offsets = self._sample_clips(total_frames)
        frame_inds = clip_offsets[:, None] + np.arange(
            self.clip_len)[None, :] * self.frame_interval
        frame_inds = np.concatenate(frame_inds)

        if self.temporal_jitter:
            perframe_offsets = np.random.randint(
                self.frame_interval, size=len(frame_inds))
            frame_inds += perframe_offsets

        frame_inds = frame_inds.reshape((-1, self.clip_len))
        if self.out_of_bound_opt == 'loop':
            frame_inds = np.mod(frame_inds, total_frames)
        elif self.out_of_bound_opt == 'repeat_last':
            safe_inds = frame_inds < total_frames
            unsafe_inds = 1 - safe_inds
            last_ind = np.max(safe_inds * frame_inds, axis=1)
            new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
            frame_inds = new_inds
        else:
            raise ValueError('Illegal out_of_bound option.')

        start_index = results['start_index']
        frame_inds = np.concatenate(frame_inds) + start_index
        results['frame_inds'] = frame_inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'frame_interval={self.frame_interval}, '
                    f'num_clips={self.num_clips}, '
                    f'temporal_jitter={self.temporal_jitter}, '
                    f'twice_sample={self.twice_sample}, '
                    f'out_of_bound_opt={self.out_of_bound_opt}, '
                    f'test_mode={self.test_mode})')
        return repr_str


# Todo:develop my sampler
@PIPELINES.register_module()
class AdaptiveSampleFrames:
    """Adaptive sample frames from the video.
    Args:
        clip_len (int): Frames of each sampled output clip.
        num_clips (int): Number of clips to be sampled. Default: 1.
        test_mode (bool): Store True when building test or validation dataset.
            Default: False.
        seed (int): The random seed used during test time. Default: 255.
    """

    def __init__(self,
                 clip_len,
                 num_clips=1,
                 test_mode=False,
                 float_ok=False,
                 p_interval=1,
                 seed=255,
                 order=False  # 是否使用增加的2行排序代码
                 ):

        self.order = order
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode
        self.float_ok = float_ok
        self.seed = seed
        self.p_interval = p_interval
        if not isinstance(p_interval, tuple):
            self.p_interval = (p_interval, p_interval)

        if self.float_ok:
            warnings.warn('When float_ok == True, there will be no loop.')

    def _get_train_clips(self, num_frames, clip_len, x, x_scores):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        allinds = []
        for clip_idx in range(self.num_clips):
            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if self.float_ok:
                interval = (num_frames - 1) / clip_len
                offsets = np.arange(clip_len) * interval
                inds = np.random.rand(clip_len) * interval + offsets
                inds = inds.astype(np.float32)
            elif num_frames < clip_len:
                start = np.random.randint(0, num_frames)
                inds = np.arange(start, start + clip_len)
            elif clip_len <= num_frames < 2 * clip_len:
                basic = np.arange(clip_len)
                inds = np.random.choice(
                    clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                inds = self._sample_merge(x, clip_len, x_scores)

            inds = inds + off
            num_frames = old_num_frames

            allinds.append(inds)

        return np.concatenate(allinds)

    def _get_test_clips(self, num_frames, clip_len, x, x_scores):
        """Uniformly sample indices for testing clips.

        Args:
            num_frames (int): The number of frames.
            clip_len (int): The length of the clip.
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        if self.float_ok:
            interval = (num_frames - 1) / clip_len
            offsets = np.arange(clip_len) * interval
            inds = np.concatenate([
                np.random.rand(clip_len) * interval + offsets
                for i in range(self.num_clips)
            ]).astype(np.float32)

        all_inds = []

        for i in range(self.num_clips):

            old_num_frames = num_frames
            pi = self.p_interval
            ratio = np.random.rand() * (pi[1] - pi[0]) + pi[0]
            num_frames = int(ratio * num_frames)
            off = np.random.randint(old_num_frames - num_frames + 1)

            if num_frames < clip_len:
                start_ind = i if num_frames < self.num_clips else i * num_frames // self.num_clips
                inds = np.arange(start_ind, start_ind + clip_len)
            elif clip_len <= num_frames < clip_len * 2:
                basic = np.arange(clip_len)
                inds = np.random.choice(clip_len + 1, num_frames - clip_len, replace=False)
                offset = np.zeros(clip_len + 1, dtype=np.int64)
                offset[inds] = 1
                offset = np.cumsum(offset)
                inds = basic + offset[:-1]
            else:
                # bids = np.array([i * num_frames // clip_len for i in range(clip_len + 1)])
                # bsize = np.diff(bids)
                # bst = bids[:clip_len]
                # offset = np.random.randint(bsize)
                # inds = bst + offset
                inds = self._sample_merge(x, clip_len, x_scores)

            all_inds.append(inds + off)
            num_frames = old_num_frames

        return np.concatenate(all_inds)

    def __call__(self, results):
        num_frames = results['total_frames']

        if self.test_mode:
            inds = self._get_test_clips(num_frames, self.clip_len, results['keypoint'], results['keypoint_score'])
        else:
            inds = self._get_train_clips(num_frames, self.clip_len, results['keypoint'], results['keypoint_score'])

        inds = np.mod(inds, num_frames)
        start_index = results['start_index']
        inds = inds + start_index

        # 22.07.27 add
        if self.order:
            for i in range(self.num_clips):
                inds[i:i+self.clip_len].sort()

        if 'keypoint' in results:
            kp = results['keypoint']
            assert num_frames == kp.shape[1]
            num_person = kp.shape[0]
            num_persons = [num_person] * num_frames
            for i in range(num_frames):
                j = num_person - 1
                while j >= 0 and np.all(np.abs(kp[j, i]) < 1e-5):
                    j -= 1
                num_persons[i] = j + 1
            transitional = [False] * num_frames
            for i in range(1, num_frames - 1):
                if num_persons[i] != num_persons[i - 1]:
                    transitional[i] = transitional[i - 1] = True
                if num_persons[i] != num_persons[i + 1]:
                    transitional[i] = transitional[i + 1] = True
            inds_int = inds.astype(np.int)
            coeff = np.array([transitional[i] for i in inds_int])
            inds = (coeff * inds_int + (1 - coeff) * inds).astype(np.float32)

        results['frame_inds'] = inds if self.float_ok else inds.astype(np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'clip_len={self.clip_len}, '
                    f'num_clips={self.num_clips}, '
                    f'test_mode={self.test_mode}, '
                    f'seed={self.seed})')
        return repr_str

    def _sample_merge(self, x, clip_len, x_scores):    # full_segment, n_segment -> num_frames = 60, clip_len = 48
        x_tmp = x.copy()
        x_scores_tmp = x_scores.copy()

        n_people, t, n_kp, xy = x_tmp.shape  # x.shape = (1, 48, 17, 2)

        # # 错误代码，没有考虑多个人的情况
        # x_1 = x_tmp.reshape(t,n_kp,xy*n_people) # x.shape = (48, 17, 2)
        # x_scores_1 = x_scores_tmp.reshape(t, n_kp)
        # # 注意：此代码没有考虑第0点的位移 88.25/91.32
        # x_1 = x_1-x_1[:,0:1,:]     # 坐标根据第0点进行调整，注意：此代码没有考虑第0点的位移

        # 改进代码，考虑了多人情况，为什么准确率反而下降了
        x = x_tmp.transpose(1, 0, 2, 3).reshape(t, n_kp*n_people, xy)
        x_scores = x_scores_tmp.transpose(1, 0, 2).reshape(t, n_kp*n_people)
        # 注意：此代码没有考虑第0点的位移 88.25/91.32
        for i in range(n_people):
            x[:, i*n_kp:(i+1)*n_kp, :] = x[:, i*n_kp:(i+1)*n_kp, :]-x[:, i*n_kp:i*n_kp+1, :]    # 坐标根据第0点进行调整，注意：此代码没有考虑第0点的位移

        # 考虑了第0点的位移
        # x[:, 1:, :] = x[:, 1:, :] - x[:, 0, :].reshape(t, 1, xy * n_people)  # 坐标根据第0点进行调整

        sampled_inds = self._slope_selection(x, clip_len, x_scores)  #

        return sampled_inds

    def _slope_selection(self, x, clip_len, x_scores):

        # MGSampler 参考代码，没有随机性，10clip的效果自然要差一些
        # def find_nearest(array, value):
        #     array = np.asarray(array)
        #     try:
        #         idx = (np.abs(array - value)).argmin()
        #         return int(idx + 1)
        #     except(ValueError):
        #         print(results['filename'])
        #
        # tmp_x = np.zeros(x.shape)
        # tmp_x[1:] = x[:-1]
        #
        # diff = x - tmp_x                             # diff.shape = (48,17,2)
        # distance = np.sqrt((diff ** 2).sum(axis=2))  # distance.shape = (48,17)
        # distance = distance*x_scores                 # 距离乘以置信度
        # distance = distance.sum(axis=1)              # distance.shape = (48,)    得到2帧之间的差异值
        #
        # accu = [0]
        # for i in range(x.shape[0]-1):
        #     accu.append(accu[-1]+distance[i+1])
        # dis_index = accu / accu[-1]
        #
        # choose_index = list()
        # if self.test_mode:
        #     for i in range(clip_len):
        #         choose_index.append(find_nearest(dis_index, 1 / (clip_len*2) + (i / clip_len)))
        #
        # return np.array(choose_index)

        # 08.03 之前的代码
        # x.shape = (48,17,2)
        tmp_x = np.zeros(x.shape)
        tmp_x[1:] = x[:-1]

        diff = x - tmp_x                             # diff.shape = (48,17,2)
        distance = np.sqrt((diff ** 2).sum(axis=2))  # distance.shape = (48,17)
        distance = distance*x_scores                 # 距离乘以置信度
        distance = distance.sum(axis=1)              # distance.shape = (48,)    得到2帧之间的差异值

        # 不采用累计分布的原始代码,按斜率来
        # inds = distance[1:].argsort()+1
        # inds = np.append(inds[-clip_len+1:],np.array([0]))  # inds.shape = (48,)
        # inds.sort()

        # 累计分布代码，不用的时候注释掉
        accu = [0]
        for i in range(x.shape[0]-1):
            accu.append(accu[-1]+distance[i+1])
        dis_index = accu[-1] / clip_len

        inds = []
        list_his = []
        for i in range(clip_len):
            list = []
            for j in range(x.shape[0]):
                if dis_index * i <= accu[j] < dis_index * (i + 1)+ 0.01:
                    list.append(j)
            if len(list) > 0:
                list_his = list
                inds.append(random.choice(list))
            else:
                inds.append(random.choice(list_his))

        inds = np.array(inds)
        return inds
