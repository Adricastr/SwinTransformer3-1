from vst import load_trained_model, real_time_video_prediction
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, num_frames, img_size = load_trained_model('best_video_swin_model.pth', device)


predictions = real_time_video_prediction(
    model, 'Project 3-1\splitVideos\id_23\S001_id23_segment_4.mp4', num_frames, img_size, device
)