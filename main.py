import matplotlib.pyplot as plt
n = [1, 2]
loss = [3, 4]
acc = [4, 7]

plt.plot(n, loss)
plt.figure(1)
plt.xlabel("n")
plt.ylabel("loss")
plt.savefig('./loss.png')
plt.figure(2)
plt.plot(n, acc)
plt.xlabel("n")
plt.ylabel("acc")
plt.savefig('./acc.png')