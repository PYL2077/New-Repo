import json
with open("/path/to/train.json", "r") as file:
	train = json.load(file)
with open("/path/to/dev.json", "r") as file:
	dev = json.load(file)
with open("/path/to/test.json", "r") as file:
	test = json.load(file)
with open("/path/to/imsitu_space.json", "r") as file:
	noun_space = json.load(file)


data = {**train, **dev, **test}

noun_total_dict = {}


total_nouns = 0

for i in data:
	for j in data[i]["frames"]:
		for k in j:
			verb = data[i]["verb"]
			noun = k
			
			if not noun in noun_total_dict:
				noun_total_dict[noun] = 0
				noun_total_dict[noun] = 0

			noun_total_dict[noun] += 1
			total_nouns += 1

# Most Frequent

# for i in range(100):
# 	noun_key = max(noun_total_dict, key=noun_total_dict.get)
# 	print(f"{noun_key}: {noun_total_dict[noun_key]} / {total_nouns}")
# 	noun_total_dict[noun_key] = 0

# Most Infrequent

for i in range(100):
	noun_key = min(noun_total_dict, key=noun_total_dict.get)
	print(f"{noun_key}: {noun_total_dict[noun_key]} / {total_nouns}")
	noun_total_dict[noun_key] = 1e11
