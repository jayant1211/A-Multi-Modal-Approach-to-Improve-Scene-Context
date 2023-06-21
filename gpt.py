import openai
import os
import re

openai.api_key = '' #put your openAPI key here

place = "bedroom"
formatted_captions = ['two people are sitting on bench in front of the water',
 'two people are sitting on bench in front of store',
 'two people are walking down the street']

description = "A boy is playing with football"
formatted_places = ['desert_road', 'ice_shelf', 'desert_sand', 'runway', 'snowfield']

prompt_base_place = "Only one answer. Print corresponding index number of correct option:  which of the following descriptions match can be most suitable for {} place: - {}".format(place,formatted_captions)
prompt_base_description = "Only one answer. Print corresponding index number of correct option: which of the following place is most probably can have a scenario in it as 'A boy is playing with football': - {}]".format(formatted_places) 


response = openai.Completion.create(
    engine="text-davinci-002", # You can select the model to use here
    prompt=prompt_base_place,
    max_tokens=50,
)

generated_text = response.choices[0].text.strip()

#print(generated_text)

numbers = re.sub('[^0-9]', '', generated_text)
print(numbers)