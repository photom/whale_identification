#/usr/bin/env python
import matplotlib.pyplot as plt

durations = []
with open('durationlist.txt') as f:
  for line in f.readlines():
    items = line.strip().split()
    if len(items) == 2:
      duration = int(items[0])
    elif len(items) == 4:
      duration = int(items[2])
      duration = duration + int(items[0]) * 1000
    else:
      print(line)
    durations.append(duration)

plt.hist(durations, bins=100)  # arguments are passed to np.histogram
plt.title("Duration Histogram with 100 bins")
plt.show()

