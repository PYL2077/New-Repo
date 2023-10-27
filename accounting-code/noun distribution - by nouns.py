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

# verb_total_dict = {}
noun_total_dict = {}


total_nouns = 0

for i in data:
	for j in data[i]["frames"]:
		for k in j:
			verb = data[i]["verb"]
			role = k
			noun = j[k]
			if noun == "": noun = "blank"
			else: noun = noun_space["nouns"][j[k]]["gloss"][0]
			

			if not noun in noun_total_dict:
				noun_total_dict[noun] = 0
				noun_total_dict[noun] = 0

			# verb_total_dict[verb] += 1
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
