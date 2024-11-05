import torch
import time
import argparse

def occupy_gpu_with_utilization(gpu_id, target_utilization=0):
    # Check if the specified GPU is available
    if not torch.cuda.is_available():
        print("No GPU available.")
        return

    # Set the desired GPU device
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using GPU: {torch.cuda.get_device_name(device)}")

    # Perform computation to occupy the specified GPU and reach the target utilization
    while True:
        with torch.cuda.device(device):
            # Increase the size of the tensors to make the computation more intense
            a = torch.randn((45000,45000), device=device)
            b = torch.randn((45000, 45000), device=device)
            c = torch.mm(a, b)

        # Sleep to avoid high CPU usage
        time.sleep(1)
        # Check the current GPU utilization
        current_utilization = torch.cuda.current_stream(device).cuda_stream
        # print(current_utilization)
        # Adjust the size of the tensors dynamically based on the utilization
        if current_utilization < target_utilization:
            a = torch.randn((45000, 45000), device=device)
            b = torch.randn((45000,45000), device=device)
            c = torch.mm(a, b)
    # while True:
    #     with torch.cuda.device(device):
    #         # Increase the size of the tensors to make the computation more intense
    #         a = torch.randn((25000,25000), device=device)
    #         b = torch.randn((25000, 25000), device=device)
    #         c = torch.mm(a, b)

    #     # Sleep to avoid high CPU usage
    #     time.sleep(1)

    #     # Check the current GPU utilization
    #     current_utilization = torch.cuda.current_stream(device).cuda_stream

    #     # Adjust the size of the tensors dynamically based on the utilization
    #     if current_utilization < target_utilization:
    #         a = torch.randn((25000, 25000), device=device)
    #         b = torch.randn((25000,25000), device=device)
    #         c = torch.mm(a, b)
if _name_ == "_main_":
    # Specify the GPU ID you want to use (e.g., 0, 1, 2, etc.)
    parser = argparse.ArgumentParser(description='Script')

    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id')


    # Specify the target GPU utilization percentage
    target_utilization_percentage = 50
    args = parser.parse_args()

    occupy_gpu_with_utilization(args.gpu_id, target_utilization_percentage)