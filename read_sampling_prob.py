import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os, subprocess
import glob
fig = plt.figure()
plt.show(block=False)
ax1 = fig.add_subplot(111)

with open("runs/cedar/2024_07_18__21_20_59__adaptive_net/1/models/Walker3DStepperEnv-v0_sampling_prob85.pkl", "rb") as fp:
	sampling_prob_list = pickle.load(fp)

sampling_prob_list = np.stack(sampling_prob_list)
print(sampling_prob_list.shape)

yaw_range = np.array([-70, 70])
heading_variation_range = np.array([0,1])

yaw_sample_size = 11
heading_variation_sample_size = 11

yaw_samples = [f"{x:.1f}" for x in np.linspace(*yaw_range, num=yaw_sample_size)]
heading_variation_samples = [f"{x:.1f}" for x in np.linspace(*heading_variation_range, num=heading_variation_sample_size)]

# frames = []
for i in range(0, len(sampling_prob_list), 100):
	ax1.clear()
	sample_cut = sampling_prob_list[i]
	tt = ax1.imshow(sample_cut)
	ax1.set_xticks(np.arange(sample_cut.shape[1]))
	ax1.set_yticks(np.arange(sample_cut.shape[0]))
	ax1.set_yticklabels(yaw_samples)
	ax1.set_xticklabels(heading_variation_samples)
	# fig.colorbar(tt)
	plt.savefig("images/file%04d.png" % i)
	# frames.append([tt])

# os.chdir("images")
# subprocess.call([
# 	'ffmpeg', '-framerate', '8', '-i', 'file%04d.png', '-r', '30', '-pix_fmt', 'yuv420p',
# 	'video_name.mp4'
# ])
# for file_name in glob.glob("*.png"):
# 	os.remove(file_name)

# sampling_sum = np.sum(sampling_prob_list[:, :, :], axis=1)
# print(sampling_sum.shape)

# for i in range(sampling_sum.shape[1]):
# 	ax1.plot(sampling_sum[:, i], label="{}".format(i-5))
# ax1.legend()
plt.show()