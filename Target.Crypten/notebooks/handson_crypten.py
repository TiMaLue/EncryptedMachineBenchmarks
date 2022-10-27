import crypten
from crypten import encoder
from examples.multiprocess_launcher import MultiProcessLauncher


def sum():
    import crypten
    import torch
    x_enc = crypten.cryptensor([1, 2, 3])
    y = 2.0
    y_enc = crypten.cryptensor(2)
    crypten.init("crypten_config.yaml")
    # crypten.mpc.provider.ttp_provider.TTPClient._init()
    r = crypten.comm.get().get_rank()

    # # Addition
    # z_enc1 = x_enc + y      # Public
    # z_enc2 = x_enc + y_enc  # Private
    # pt = z_enc1.get_plain_text()
    # pt1= z_enc2.get_plain_text()
    # if r == 0:
    #     crypten.print("\nPublic  addition:", pt)
    #     crypten.print("Private addition:", pt1)


    # # Subtraction
    # z_enc1 = x_enc - y      # Public
    # z_enc2 = x_enc - y_enc  # Private
    # crypten.print("\nPublic  subtraction:", z_enc1.get_plain_text())
    # print("Private subtraction:", z_enc2.get_plain_text())

    # Multiplication
    z_enc1 = x_enc * y      # Public
    z_enc2 = x_enc * y_enc  # Private
    pt1 = z_enc1.get_plain_text()
    pt2 = z_enc2.get_plain_text()
    if r == 0:
        print("Public  multiplication:", pt1)
        print("Private multiplication:", pt2)

    # # Division
    # z_enc1 = x_enc / y      # Public
    # z_enc2 = x_enc / y_enc  # Private
    # if r == 0:
    #     print("\nPublic  division:", z_enc1.get_plain_text())
    #     print("Private division:", z_enc2.get_plain_text())



if __name__ == "__main__":
    args = []
    entry_point = sum
    world_size = 3
    encoder
    launcher = MultiProcessLauncher(
        world_size, entry_point
    )
    launcher.start()
    launcher.join()
    launcher.terminate()