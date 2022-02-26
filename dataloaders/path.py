# -*- coding: utf-8 -*-
"""
 @Time    : 2019/1/21 22:07
 @Author  : Wang Xin
 @Email   : wangxin_buaa@163.com
"""


class Path(object):
    @staticmethod
    def db_root_dir(database):
        if database == 'nyu':
            return '/mnt/hdd/shared_datasets/nyudepthv2'
        elif database == 'kitti':
            return '/mnt/hdd/shared_datasets/kitti'
        elif database == 'floorplan3d':
            return '/mnt/hdd/shared_datasets/floorplan3d'
        elif database == 'Structured3D':
            return '/mnt/hdd/shared_datasets/Structured3D'
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError
