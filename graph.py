import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def onclick(event):
    print("button=%d, x=%d, y=%d, xdata=%f, ydata=%f" % (
        event.button, event.x, event.y, event.xdata, event.ydata))


img = mpimg.imread("train.jpg")
ax = plt.imshow(img)
fig = ax.get_figure()
cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
