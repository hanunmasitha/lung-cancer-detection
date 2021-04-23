import os
import glob

print("a")
#for address in os.listdir("/koding/python/TA/test/tumor"):
 #   print(address)

for address in glob.glob("/koding/python/TA/test/tumor/*"):
    print(address)

result = {
  "Kanker": 0,
  "Normal": 0,
  "Tumor": 0
}
result_key = list(result)
for key in range(len(result_key)):
    print(key)

    print(result_key[key])