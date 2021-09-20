def world_to_camera(X, K_EX):
    K_inv = np.linalg.inv(K_EX)
    return np.matmul(K_inv[:3,:3], X.T).T + K_inv[:3,3]
    
def project_to_2d(X, camera_params):
    """
    Project 3D points to 2D using the Human3.6M camera projection function.
    This is a differentiable and batched reimplementation of the original MATLAB script.

    Arguments:
    X -- 3D points in *camera space* to transform (N, *, 3)
    camera_params -- intrinsic parameteres (N, 2+2+3+2=9)
    focal length / principal point / radial_distortion / tangential_distortion
    """
    assert X.shape[-1] == 3
    assert len(camera_params.shape) == 2
    assert camera_params.shape[-1] == 9
    assert X.shape[0] == camera_params.shape[0]

    while len(camera_params.shape) < len(X.shape):
        camera_params = camera_params.unsqueeze(1)

    f = camera_params[..., :2] #focal lendgth
    c = camera_params[..., 2:4] # center principal point
    k = camera_params[..., 4:7]
    p = camera_params[..., 7:]

    XX = torch.clamp(X[..., :2] / X[..., 2:], min=-1, max=1)
    r2 = torch.sum(XX[..., :2]**2, dim=len(XX.shape)-1, keepdim=True)

    radial = 1 + torch.sum(k * torch.cat((r2, r2**2, r2**3), dim=len(r2.shape)-1), dim=len(r2.shape)-1, keepdim=True)
    tan = torch.sum(p*XX, dim=len(XX.shape)-1, keepdim=True)

    XXX = XX*(radial + tan) + p*r2

    return f*XXX + c

def render_animation(keypoints, poses, fps, bitrate, azim, output, viewport, cloud=None,
                     limit=-1, downsample=1, size=5, input_video_path=None, input_video_skip=0, frame_num=None, key=None):
    """
    TODO
    Render an animation. The supported output modes are:
     -- 'interactive': display an interactive figure
                       (also works on notebooks if associated with %matplotlib inline)
     -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
     -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
     -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
    """

    path = './data/%s/%s' % (dirs, output.split('.')[0])
    if not os.path.exists(path):
        os.mkdir(path)
    # plt.ioff()
    # figsize = (10, 5)
    fig = plt.figure(figsize=(size*(1 + len(poses)), size))
    # 2D
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)  # (1,2,1)
    ax_in.get_xaxis().set_visible(False)
    ax_in.get_yaxis().set_visible(False)
    ax_in.set_axis_off()
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    # 0, ('Reconstruction', 3d kp)
    for index, (title, data) in enumerate(poses.items()):
        # 3D
        ax = fig.add_subplot(1, 1 + len(poses), index+2,
                             projection='3d')  # (1,2,2)
        ax.view_init(elev=15., azim=azim)
        # set 长度范围
        ax.set_xlim3d([-radius/2, radius/2])
        ax.set_ylim3d([-radius/2, radius/2])
        ax.set_zlim3d([0, radius])
        # ax.auto_scale_xyz([-radius/2, radius/2], [0, radius], [-radius/2, radius/2])
        # axisEqual3D(ax)
        ax.dist = 9  # 视角距离
        ax.set_title(title)  # , pad=35
        ax_3d.append(ax)
        lines_3d.append([])

        # lxy add
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # 轨迹 is base on position 0
        trajectories.append(data[:, 0])  # only add x,y not z
    poses = list(poses.values())

    # 设置2、3的显示
    if cloud is not None:
        for n, ax in enumerate(ax_3d):
            if n == 0:
                norm = plt.Normalize(min(cloud[2]), max(cloud[2]))
                ax.scatter(*cloud, c=norm(cloud[2]),marker='.', s=5, linewidth=0, alpha=0.3, cmap='viridis')
            elif n == 1:
                cloud_in = cloud[:, cloud[2] >= 0.1]
                norm = plt.Normalize(min(cloud_in[2]), max(cloud_in[2]))
                ax.scatter(*cloud_in, c=norm(cloud_in[2]),marker='.', s=3, alpha=0.3, cmap='Greys')
                ax.view_init(elev=90, azim=0)
            elif n == 2:
                cloud_in = cloud[:, cloud[2] < 0.1]
                norm = plt.Normalize(min(cloud_in[2]), max(cloud_in[2]))
                ax.scatter(*cloud_in, c=norm(cloud_in[2]),marker='.', s=3, alpha=0.3, cmap='Greys')
                ax.view_init(elev=0, azim=0)

            if n > 0:
                ax.dist = 7
                middle_y = np.mean(trajectories[n][:, 1])  # y轴的中点
                center = np.mean(cloud_in, -1)  # x的中心
                mb = max(trajectories[n][:, 1]) - min(trajectories[n][:, 1])   # 以轨迹的Y轴为可视化的最大长度
                ax.set_xlim3d([-mb/2 + center[0], mb/2 + center[0]])
                ax.set_ylim3d([-mb/2 + middle_y, mb/2 + middle_y])
                ax.set_zlim3d([-mb/2, mb/2])
    
    # Decode video
    if input_video_path is None:
        # Black background
        all_frames = np.zeros(
            (keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # Load video using ffmpeg
        # 根据kpt长度，决定帧的长度
        all_frames = []
        for f in read_video(input_video_path, skip=input_video_skip):
            all_frames.append(f)
        # effective_length = min(keypoints.shape[0], len(all_frames))
        # all_frames = all_frames[:effective_length]

    kpts = keypoints
    initialized = False
    image = None
    lines = []
    numbers = []
    points = None

    if limit < 1:
        limit = len(frame_num)
    else:
        limit = min(limit, len(frame_num))

    vis_enhence = True

    def update_video(i):
        # if i < frame_num[15] or i > frame_num[-15]:
        #     return
        nonlocal initialized, image, lines, points, numbers
        for num in numbers:
            num.remove()
        numbers.clear()
        for n, ax in enumerate(ax_3d):  # 只有1个
            if i > 0:  # 绘制轨迹
                dt = trajectories[n][i-1:i+1]
                # if np.linalg.norm(dt[0] - dt[1]) < 1:
                    # ax.plot(*dt.T, c='red', linewidth=2, alpha=0.5)
            if n == 0:
                ax.set_xlim3d([-radius/2 + 1.2, radius/2 + 1.2])
                ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
                ax.set_ylim3d([-radius/2 + trajectories[n][i, 1],radius/2 + trajectories[n][i, 1]])
            ax.set_zlim3d([-radius/2 + trajectories[n][i, 2],
                           radius/2 + trajectories[n][i, 2]])
            # axisEqual3D(ax)

        # Update 2D poses
        if not initialized:
            image = ax_in.imshow(all_frames[frame_num[i]], aspect='equal')
            # 画图2D
            points = ax_in.scatter(*kpts[i].T, 2, color='pink', edgecolors = 'white', zorder = 10)
            initialized = True
        else:
            image.set_data(all_frames[frame_num[i]])
            points.set_offsets(keypoints[i])

        if i % 50 == 0 and vis_enhence:
            plt.savefig(
                path + '/' + str(frame_num[i]), dpi=100, bbox_inches='tight')
        print(
            'finish one frame\t {}/{}'.format(frame_num[i], frame_num[-1]), end='\r')

    fig.tight_layout()

    anim = FuncAnimation(fig, update_video, frames=np.arange(
        0, limit), interval=1000/fps, repeat=False)
        
    save_path = os.path.join(path, "%s_%d_%s_%s" %(key, keypoints.shape[0], dirs, output))
    
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(save_path, writer=writer)
    elif output.endswith('.gif'):
        # anim.save(output, dpi=80, writer='imagemagick')
        anim.save(save_path, dpi=80, writer='imagemagick')
    else:
        raise ValueError(
            'Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()


def _get_overlap(lidar, rot_data, lidar_key, mocap_key):
    lidar_key = lidar_key - int(lidar[0,0]) 
    mocap_length = rot_data.shape[0]
    lidar_start = 0
    lidar_end = int(lidar[-1,0]) - int(lidar[0,0]) 
    mocap_start = mocap_key - (lidar_key - lidar_start) * frame_scale
    mocap_end = mocap_key + (lidar_end - lidar_key) * frame_scale

    if (mocap_start) < 0:
        lidar_start = lidar_key - (mocap_key // frame_scale)
        mocap_start = mocap_key % frame_scale 
    if (mocap_end) > mocap_length - 1:
        lidar_end = lidar_key + (mocap_length - 1 - mocap_key) // frame_scale
        mocap_end = mocap_length - 1 - (mocap_length - 1 - mocap_key) % frame_scale
    print('LiDAR start frame to end: ', lidar_start, lidar_end)
    print('Mocap start frame to end: ', mocap_start, mocap_end)
    return lidar_start, lidar_end, mocap_start, mocap_end