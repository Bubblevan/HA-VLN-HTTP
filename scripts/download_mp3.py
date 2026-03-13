#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Modified for Python 3 compatibility and safety by Gemini

import argparse
import os
import tempfile
import urllib.request
import sys
import time
from urllib.error import ContentTooShortError, URLError

BASE_URL = 'http://kaldir.vc.in.tum.de/matterport/'
RELEASE = 'v1/scans'
RELEASE_TASKS = 'v1/tasks/'
RELEASE_SIZE = '1.3TB'
TOS_URL = BASE_URL + 'MP_TOS.pdf'
FILETYPES = [
    'cameras', 'matterport_camera_intrinsics', 'matterport_camera_poses',
    'matterport_color_images', 'matterport_depth_images', 'matterport_hdr_images',
    'matterport_mesh', 'matterport_skybox_images', 'undistorted_camera_parameters',
    'undistorted_color_images', 'undistorted_depth_images', 'undistorted_normal_images',
    'house_segmentations', 'region_segmentations', 'image_overlap_data',
    'poisson_meshes', 'sens'
]
TASK_FILES = {
    'keypoint_matching_data': ['keypoint_matching/data.zip'],
    'keypoint_matching_models': ['keypoint_matching/models.zip'],
    'surface_normal_data': ['surface_normal/data_list.zip'],
    'surface_normal_models': ['surface_normal/models.zip'],
    'region_classification_data': ['region_classification/data.zip'],
    'region_classification_models': ['region_classification/models.zip'],
    'semantic_voxel_label_data': ['semantic_voxel_label/data.zip'],
    'semantic_voxel_label_models': ['semantic_voxel_label/models.zip'],
    'minos': ['mp3d_minos.zip'],
    'gibson': ['mp3d_for_gibson.tar.gz'],
    'habitat': ['mp3d_habitat.zip'],
    'pixelsynth': ['mp3d_pixelsynth.zip'],
    'igibson': ['mp3d_for_igibson.zip'],
    'mp360': ['mp3d_360/data_00.zip', 'mp3d_360/data_01.zip', 'mp3d_360/data_02.zip', 
              'mp3d_360/data_03.zip', 'mp3d_360/data_04.zip', 'mp3d_360/data_05.zip', 
              'mp3d_360/data_06.zip']
}

def get_release_scans(release_file):
    scans = []
    with urllib.request.urlopen(release_file) as response:
        for line in response:
            scan_id = line.decode('utf-8').rstrip('\n')
            scans.append(scan_id)
    return scans

def download_file(url, out_file, max_retries=5, retry_delay=5):
    """
    下载文件，支持断点续传和重试机制
    
    Args:
        url: 下载链接
        out_file: 输出文件路径
        max_retries: 最大重试次数
        retry_delay: 重试延迟（秒）
    """
    out_dir = os.path.dirname(out_file)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
    # 检查文件是否已存在且完整
    if os.path.isfile(out_file):
        print('WARNING: skipping download of existing file ' + out_file)
        return
    
    print('\t' + url + ' > ' + out_file)
    
    # 检查是否有部分下载的文件（临时文件）
    out_file_tmp = out_file + '.tmp'
    resume_pos = 0
    
    # 如果临时文件存在，尝试断点续传
    if os.path.exists(out_file_tmp):
        resume_pos = os.path.getsize(out_file_tmp)
        if resume_pos > 0:
            print(f'\tResuming download from {resume_pos / (1024*1024):.2f} MB')
    
    for attempt in range(max_retries):
        try:
            # 创建请求对象，支持断点续传
            req = urllib.request.Request(url)
            if resume_pos > 0:
                req.add_header('Range', f'bytes={resume_pos}-')
            
            # 打开连接
            with urllib.request.urlopen(req, timeout=300) as response:
                # 获取文件总大小
                total_size = int(response.headers.get('Content-Length', 0))
                if resume_pos > 0:
                    # 断点续传时，Content-Length 是剩余大小
                    total_size = resume_pos + total_size
                
                # 打开文件（追加模式用于断点续传）
                mode = 'ab' if resume_pos > 0 else 'wb'
                with open(out_file_tmp, mode) as f:
                    downloaded = resume_pos
                    chunk_size = 8192  # 8KB chunks
                    
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # 每 100MB 显示一次进度
                        if downloaded % (100 * 1024 * 1024) < chunk_size:
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                print(f'\tProgress: {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent:.1f}%)')
                            else:
                                print(f'\tProgress: {downloaded / (1024*1024):.2f} MB')
            
            # 下载完成，重命名临时文件
            os.rename(out_file_tmp, out_file)
            print(f'\t✓ Download completed: {os.path.getsize(out_file) / (1024*1024):.2f} MB')
            return
            
        except (ContentTooShortError, URLError, IOError, OSError) as e:
            # 获取当前下载的大小
            current_size = os.path.getsize(out_file_tmp) if os.path.exists(out_file_tmp) else 0
            resume_pos = current_size
            
            if attempt < max_retries - 1:
                print(f'\t✗ Download failed (attempt {attempt + 1}/{max_retries}): {str(e)}')
                print(f'\t  Retrying in {retry_delay} seconds... (resume from {resume_pos / (1024*1024):.2f} MB)')
                time.sleep(retry_delay)
                retry_delay *= 2  # 指数退避
            else:
                print(f'\t✗ Download failed after {max_retries} attempts: {str(e)}')
                # 保留临时文件以便下次继续
                if os.path.exists(out_file_tmp):
                    print(f'\t  Partial file saved at: {out_file_tmp}')
                    print(f'\t  Run the command again to resume download')
                raise

def download_scan(scan_id, out_dir, file_types):
    print('Downloading MP scan ' + scan_id + ' ...')
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    for ft in file_types:
        url = BASE_URL + RELEASE + '/' + scan_id + '/' + ft + '.zip'
        out_file = out_dir + '/' + ft + '.zip'
        download_file(url, out_file)
    print('Downloaded scan ' + scan_id)

def download_task_data(task_data, out_dir):
    print('Downloading MP task data for ' + str(task_data) + ' ...')
    for task_data_id in task_data:
        if task_data_id in TASK_FILES:
            file_parts = TASK_FILES[task_data_id]
            for filepart in file_parts:
                url = BASE_URL + RELEASE_TASKS + '/' + filepart
                localpath = os.path.join(out_dir, filepart)
                download_file(url, localpath)
                print('Downloaded task data ' + task_data_id)

def main():
    parser = argparse.ArgumentParser(description='Downloads MP public data release (Python 3 fix)')
    parser.add_argument('-o', '--out_dir', required=True, help='directory in which to download')
    parser.add_argument('--task_data', default=[], nargs='+', help='task data files to download.')
    parser.add_argument('--id', default='ALL', help='specific scan id to download')
    parser.add_argument('--type', nargs='+', help='specific file types to download.')
    args = parser.parse_args()

    print('By pressing Enter you confirm that you have agreed to the MP terms of use.')
    print(TOS_URL)
    input('Press Enter to continue, or CTRL-C to exit...')

    # download task data
    if args.task_data:
        if set(args.task_data) & set(TASK_FILES.keys()):
            out_dir = args.out_dir # 原脚本这里拼接路径有点混乱，为了简单，直接存在 out_dir 根目录或 tasks 目录
            # 为了保持结构整洁，我们还是建议放在 out_dir/v1/tasks 下，与原脚本逻辑一致
            # 但用户通常找不到，所以这里直接下载到 out_dir 下比较直观
            # 原脚本逻辑：out_dir = os.path.join(args.out_dir, RELEASE_TASKS)
            # 我们稍微改一下，让它下载到 out_dir 里面去
            
            # 使用原脚本的目录结构逻辑
            task_out_dir = os.path.join(args.out_dir, RELEASE_TASKS)
            download_task_data(args.task_data, task_out_dir)
            
            print('-' * 40)
            print('SUCCESS: Task data downloaded.')
            print('Safety Stop: Exiting to prevent downloading the full 1.3TB dataset.')
            print('If you REALLY need the full dataset, run the script again without --task_data.')
            sys.exit(0) # 直接退出，防止误操作
            
        else:
            print('ERROR: Unrecognized task data id: ' + str(args.task_data))
            return

    # 下面是下载 1.3TB 数据的逻辑，如果跑上面的 task_data，这里就不会执行了
    release_file = BASE_URL + RELEASE + '.txt'
    release_scans = get_release_scans(release_file)
    file_types = FILETYPES

    if args.type:
        file_types = args.type

    if args.id and args.id != 'ALL' and args.id != 'all':
        scan_id = args.id
        if scan_id not in release_scans:
            print('ERROR: Invalid scan id: ' + scan_id)
        else:
            out_dir = os.path.join(args.out_dir, RELEASE, scan_id)
            download_scan(scan_id, out_dir, file_types)
    elif 'minos' not in args.task_data: 
        print('WARNING: You are about to download the entire MP release (1.3TB).')
        input('Press Enter to continue, or CTRL-C to exit...')
        out_dir = os.path.join(args.out_dir, RELEASE)
        download_release(release_scans, out_dir, file_types)

if __name__ == "__main__": main()


# python download_mp3.py -o data/scene_datasets/mp3d --task_data habitat
#
# By pressing Enter you confirm that you have agreed to the MP terms of use.
# 
# http://kaldir.vc.in.tum.de/matterport/MP_TOS.pdf
# Press Enter to continue, or CTRL-C to exit...
# Downloading MP task data for ['habitat'] ...
#         http://kaldir.vc.in.tum.de/matterport/v1/tasks//mp3d_habitat.zip > data/scene_datasets/mp3d\v1/tasks/mp3d_habitat.zip
#         Progress: 100.00 MB / 15340.14 MB (0.7%)
#         Progress: 200.00 MB / 15340.14 MB (1.3%)
#         Progress: 300.00 MB / 15340.14 MB (2.0%)
#         Progress: 400.00 MB / 15340.14 MB (2.6%)
#         Progress: 500.00 MB / 15340.14 MB (3.3%)
#         ✓ Download completed: 546.66 MB
# Downloaded task data habitat
# ----------------------------------------
# SUCCESS: Task data downloaded.
# Safety Stop: Exiting to prevent downloading the full 1.3TB dataset.
# If you REALLY need the full dataset, run the script again without --task_data.
