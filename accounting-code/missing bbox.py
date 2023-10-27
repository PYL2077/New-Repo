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
verb_missing_dict = {}
verb_total_dict = {}
noun_missing_dict = {}
noun_total_dict = {}


missing_bbox = 0
total_bbox = 0

for i in data:
	for j in data[i]["bb"]:
		print(data[i]["bb"])
		# verb = data[i]["verb"]
		# noun = j
		# if not verb in verb_missing_dict:
		# 	verb_missing_dict[verb] = 0
		# 	verb_total_dict[verb] = 0
		# if not noun in noun_missing_dict:
		# 	noun_missing_dict[noun] = 0
		# 	noun_total_dict[noun] = 0

		# if data[i]["bb"][j][0] == -1:
		# 	missing_bbox += 1
		# 	verb_missing_dict[verb] += 1
		# 	noun_missing_dict[noun] += 1

		# verb_total_dict[verb] += 1
		# noun_total_dict[noun] += 1
		# total_bbox += 1

"""		
print(missing_bbox)
print(total_bbox)

for i in range(100):
	verb_key = max(verb_missing_dict, key=verb_missing_dict.get)
	print(f"{verb_key}: {verb_missing_dict[verb_key]} / {verb_total_dict[verb_key]}")
	verb_missing_dict[verb_key] = 0

print("-------\n\n")

for i in range(100):
	noun_key = max(noun_missing_dict, key=noun_missing_dict.get)
	print(f"{noun_key}: {noun_missing_dict[noun_key]} / {noun_total_dict[noun_key]}")
	noun_missing_dict[noun_key] = 0
"""