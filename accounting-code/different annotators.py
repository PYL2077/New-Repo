import json
with open("/path/to/train.json", "r") as file:
	train = json.load(file)
with open("/path/to/dev.json", "r") as file:
	dev = json.load(file)
with open("/path/to/test.json", "r") as file:
	test = json.load(file)
with open("/path/to/imsitu_space.json", "r") as file:
	noun_space = json.load(file)

# for key in noun_space["nouns"]:
# 	print(noun_space["nouns"][key]["gloss"][0])
# 	print("\n")
	# for i in noun_space["nouns"][key]["gloss"]:
	# 	print(i)


data = {**train, **dev, **test}

verb_all_diff = {}
verb_two_diff = {}
noun_all_diff = {}
noun_two_diff = {}


total_nouns = 0
three_same = 0
two_same = 0
one_same = 0

for i in data:
	for k in data[i]["frames"][0]:
		verb = data[i]["verb"]
		noun = k
		a = data[i]["frames"][0][k]
		b = data[i]["frames"][1][k]
		c = data[i]["frames"][2][k]
		if verb not in verb_all_diff.keys():
			verb_all_diff[verb] = 0
			verb_two_diff[verb] = 0
		if noun not in noun_all_diff.keys():
			noun_all_diff[noun] = 0
			noun_two_diff[noun] = 0
		if a==b and b==c:
			three_same += 1
		elif a==b or b==c or c==a:
			two_same += 1
			verb_two_diff[verb] += 1
			noun_two_diff[noun] += 1
		else: 
			one_same += 1
			verb_two_diff[verb] += 1
			noun_two_diff[noun] += 1
			verb_all_diff[verb] += 1
			noun_all_diff[noun] += 1
		total_nouns += 1

print(three_same)
print(two_same)
print(one_same)
print(total_nouns)


# for i in range(100):
# 	noun_key = max(verb_two_diff, key=verb_two_diff.get)
# 	print(f"{noun_key}: {verb_two_diff[noun_key]} / {total_nouns}")
# 	verb_two_diff[noun_key] = 0

# print("\n------\n")

# for i in range(100):
# 	noun_key = max(verb_all_diff, key=verb_all_diff.get)
# 	print(f"{noun_key}: {verb_all_diff[noun_key]} / {total_nouns}")
# 	verb_all_diff[noun_key] = 0


for i in range(100):
	noun_key = max(noun_two_diff, key=noun_two_diff.get)
	print(f"{noun_key}: {noun_two_diff[noun_key]} / {total_nouns}")
	noun_two_diff[noun_key] = 0

print("\n------\n")

for i in range(100):
	noun_key = max(noun_all_diff, key=noun_all_diff.get)
	print(f"{noun_key}: {noun_all_diff[noun_key]} / {total_nouns}")
	noun_all_diff[noun_key] = 0
