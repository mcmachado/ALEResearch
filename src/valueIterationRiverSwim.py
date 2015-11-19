import sys

V = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

delta = 100.0
gamma = 0.95

while(delta > 0.1):
	delta = 0

	v = V[5]
	V[5] = 0.3 * (10000 + gamma * V[5]) + 0.7 * (gamma * V[4])
	delta = max(delta, abs(v - V[5]))

	v = V[4]
	V[4] = 0.3 * (gamma * V[5]) + 0.1 * (gamma * V[4]) + 0.6 * (gamma * V[3])
	delta = max(delta, abs(v - V[4]))

	v = V[3]
	V[3] = 0.3 * (gamma * V[4]) + 0.1 * (gamma * V[3]) + 0.6 * (gamma * V[2])
	delta = max(delta, abs(v - V[3]))

	v = V[2]
	V[2] = 0.3 * (gamma * V[3]) + 0.1 * (gamma * V[2]) + 0.6 * (gamma * V[1])
	delta = max(delta, abs(v - V[2]))

	v = V[1]
	V[1] = 0.3 * (gamma * V[2]) + 0.1 * (gamma * V[1]) + 0.6 * (gamma * V[0])
	delta = max(delta, abs(v - V[1]))

	v = V[0]
	V[0] = 0.3 * (gamma * V[1]) + 0.7 * (gamma * V[0])
	delta = max(delta, abs(v - V[0]))

print V[0], V[1], V[2], V[3], V[4], V[5]