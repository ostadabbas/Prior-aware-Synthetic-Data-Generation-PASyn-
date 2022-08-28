from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def read_tensorboard_data(tensorboard_path, val_name):

    ea = event_accumulator.EventAccumulator(tensorboard_path)
    ea.Reload()
    print(ea.scalars.Keys())
    val = ea.scalars.Items(val_name)
    return val

def draw_plt(val, val_name):

    plt.figure()
    plt.plot([i.step for i in val], [j.value for j in val], label=val_name)
    """x axis is step，num of iteration
    y axis is value"""
    plt.xlabel('step')
    plt.ylabel(val_name)
    plt.show()

if __name__ == "__main__":
    tensorboard_path = './result/training/1/tensorboard/version_0/events.out.tfevents.1644395799.drow.7757.0'
    val_name = 'lr-Adam'
    val = read_tensorboard_data(tensorboard_path, val_name)
    draw_plt(val, val_name)