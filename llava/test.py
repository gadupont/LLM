import replicate
import os




output = replicate.run(
    "yorickvp/llava-13b:80537f9eead1a5bfa72d5ac6ea6414379be41d4d4f6679fd776e9535d1eb58bb",
    input={
        "image": "https://replicate.delivery/pbxt/KRULC43USWlEx4ZNkXltJqvYaHpEx2uJ4IyUQPRPwYb8SzPf/view.jpg",
        "top_p": 1,
        "prompt": "Are you allowed to swim here?",
        "max_tokens": 1024,
        "temperature": 0.2
    }
)

# The yorickvp/llava-13b model can stream output as it's running.
# The predict method returns an iterator, and you can iterate over that output.
for item in output:
    # https://replicate.com/yorickvp/llava-13b/api#output-schema
    print(item, end="")
