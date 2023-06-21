import openai
import os
import re

openai.api_key = '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'

def execute_caption_base(best_caption,all_places):
    print("Using Caption as Base")
    formatted_descriptions = ", ".join([f"{i+1}. {desc}" for i, desc in enumerate(all_places)])
    prompt_base_caption = "Only one answer. Print corresponding order number of correct option: which of the following place is most probably can have a scenario in it as {} : - {}. If none, return 1".format(best_caption,formatted_descriptions) 
    print(prompt_base_caption)

    response = openai.Completion.create(
        engine="text-davinci-002", # You can select the model to use here
        prompt=prompt_base_caption,
        max_tokens=50,
    )
    generated_text = response.choices[0].text.strip()
    number = re.sub('[^0-9]', '', generated_text)
    return number

def execute_scene_base(best_place, all_captions):
    print("Using Place as Base")
    formatted_descriptions = ", ".join([f"{i+1}. {desc}" for i, desc in enumerate(all_captions)])
    prompt_base_scene = "Only one answer. Print corresponding order number of correct option:  which of the following descriptions match can be most suitable at {} place: - {}. If none, return 1.".format(best_place,formatted_descriptions)
    
    print(prompt_base_scene)
    response = openai.Completion.create(
        engine="text-davinci-002", # You can select the model to use here
        prompt=prompt_base_scene,
        max_tokens=50,
    )
    generated_text = response.choices[0].text.strip()
    print(generated_text)
    number = re.sub('[^0-9]', '', generated_text)
    return number