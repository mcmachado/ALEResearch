import matplotlib.pylab as plt

r10_avg = [11.6, 8.7, 6.0, 1.3]
r10_std = [5.1, 6.4, 2.1, 0.6]
r20_avg = [9.6, 6.5, 5.5, 1.0]
r20_std = [2.4, 4.3, 1.6, 0.3]
r30_avg = [9.2, 5.1, 5.5, 1.3]
r30_std = [2.1, 2.2, 1.8, 0.3]
r40_avg = [9.2, 5.2, 5.9, 0.9]
r40_std = [1.8, 2.0, 1.5, 0.2]
r50_avg = [9.1, 5.1, 5.5, 1.1]
r50_std = [1.7, 2.0, 1.4, 0.2]
r70_avg = [8.9, 4.8, 5.8, 0.8]
r70_std = [1.8, 1.7, 1.2, 0.1]

x = [2, 3, 4, 5]

plt.xlim([1.8,5.2])
plt.xlabel('Log. # Episodes')
plt.ylabel('Avg. Return')
plt.title(r'Comparison between $R_{max}$ influence in a 10x10 grid')

plt.plot(x, r10_avg, label=r'$R_{max} = 10$')
plt.plot(x, r20_avg, label=r'$R_{max} = 20$')
plt.plot(x, r30_avg, label=r'$R_{max} = 30$')
plt.plot(x, r40_avg, label=r'$R_{max} = 40$')
plt.plot(x, r50_avg, label=r'$R_{max} = 50$')
plt.plot(x, r70_avg, label=r'$R_{max} = 70$')


plt.legend()
plt.savefig('rmax.pdf')
