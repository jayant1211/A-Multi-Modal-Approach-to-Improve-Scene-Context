import os
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.enable_eager_execution()

import cv2
import numpy as np

model_path = 'models/pre-trained/'
names = ['airfield', 'airplane_cabin', 'airport_terminal', 'alcove', 'alley', 'amphitheater', 'amusement_arcade', 'amusement_park', 'apartment_building_outdoor', 'aquarium', 'aqueduct', 'arcade', 'arch', 'archaelogical_excavation', 'archive', 'arena_hockey', 'arena_performance', 'arena_rodeo', 'army_base', 'art_gallery', 
         'art_school', 'art_studio', 'artists_loft', 'assembly_line', 'athletic_field_outdoor', 'atrium_public', 'attic', 'auditorium', 'auto_factory', 'auto_showroom', 'badlands', 'bakery_shop', 'balcony_exterior', 'balcony_interior', 'ball_pit', 'ballroom', 'bamboo_forest', 'bank_vault', 'banquet_hall', 'bar', 'barn', 
         'barndoor', 'baseball_field', 'basement', 'basketball_court_indoor', 'bathroom', 'bazaar_indoor', 'bazaar_outdoor', 'beach', 'beach_house', 'beauty_salon', 'bedchamber', 'bedroom', 'beer_garden', 'beer_hall', 'berth', 'biology_laboratory', 'boardwalk', 'boat_deck', 'boathouse', 'bookstore', 'booth_indoor', 'botanical_garden', 'bow_window_indoor', 
         'bowling_alley', 'boxing_ring', 'bridge', 'building_facade', 'bullring', 'burial_chamber', 'bus_interior', 'bus_station_indoor', 'butchers_shop', 'butte', 'cabin_outdoor', 'cafeteria', 'campsite', 'campus', 'canal_natural', 'canal_urban', 'candy_store', 'canyon', 'car_interior', 'carrousel', 'castle', 'catacomb', 'cemetery', 'chalet', 'chemistry_lab', 'childs_room', 'church_indoor', 'church_outdoor', 'classroom', 'clean_room', 'cliff', 'closet', 'clothing_store', 'coast', 'cockpit', 'coffee_shop', 'computer_room', 'conference_center', 'conference_room', 'construction_site', 'corn_field', 'corral', 'corridor', 'cottage', 'courthouse', 'courtyard', 'creek', 'crevasse', 'crosswalk', 'dam', 'delicatessen', 'department_store', 'desert_sand', 'desert_vegetation', 'desert_road', 'diner_outdoor', 'dining_hall', 'dining_room', 'discotheque', 'doorway_outdoor', 'dorm_room', 'downtown', 'dressing_room', 'driveway', 'drugstore', 'elevator_door', 'elevator_lobby', 'elevator_shaft', 'embassy', 'engine_room', 'entrance_hall', 'escalator_indoor', 'excavation', 'fabric_store', 'farm', 'fastfood_restaurant', 'field_cultivated', 'field_wild', 'field_road', 'fire_escape', 'fire_station', 'fishpond', 'flea_market_indoor', 'florist_shop_indoor', 'food_court', 'football_field', 'forest_broadleaf', 'forest_path', 'forest_road', 'formal_garden', 'fountain', 'galley', 'garage_indoor', 'garage_outdoor', 'gas_station', 'gazebo_exterior', 'general_store_indoor', 'general_store_outdoor', 'gift_shop', 'glacier', 'golf_course', 'greenhouse_indoor', 'greenhouse_outdoor', 'grotto', 'gymnasium_indoor', 'hangar_indoor', 'hangar_outdoor', 'harbor', 'hardware_store', 'hayfield', 'heliport', 'highway', 'home_office', 'home_theater', 'hospital', 'hospital_room', 'hot_spring', 'hotel_outdoor', 'hotel_room', 'house', 'hunting_lodge_outdoor', 'ice_cream_parlor', 'ice_floe', 'ice_shelf', 'ice_skating_rink_indoor', 'ice_skating_rink_outdoor', 'iceberg', 'igloo', 'industrial_area', 'inn_outdoor', 'islet', 'jacuzzi_indoor', 'jail_cell', 'japanese_garden', 'jewelry_shop', 'junkyard', 'kasbah', 'kennel_outdoor', 'kindergarden_classroom', 'kitchen', 'lagoon', 'lake_natural', 'landfill', 'landing_deck', 'laundromat', 'lawn', 'lecture_room', 'legislative_chamber', 'library_indoor', 'library_outdoor', 'lighthouse', 'living_room', 'loading_dock', 'lobby', 'lock_chamber', 'locker_room', 'mansion', 'manufactured_home', 'market_indoor', 'market_outdoor', 'marsh', 'martial_arts_gym', 'mausoleum', 'medina', 'mezzanine', 'moat_water', 'mosque_outdoor', 'motel', 'mountain', 'mountain_path', 'mountain_snowy', 'movie_theater_indoor', 'museum_indoor', 'museum_outdoor', 'music_studio', 'natural_history_museum', 'nursery', 'nursing_home', 'oast_house', 'ocean', 'office', 'office_building', 'office_cubicles', 'oilrig', 'operating_room', 'orchard', 'orchestra_pit', 'pagoda', 'palace', 'pantry', 'park', 'parking_garage_indoor', 'parking_garage_outdoor', 'parking_lot', 'pasture', 'patio', 'pavilion', 'pet_shop', 'pharmacy', 'phone_booth', 'physics_laboratory', 'picnic_area', 'pier', 'pizzeria', 'playground', 'playroom', 'plaza', 'pond', 'porch', 'promenade', 'pub_indoor', 'racecourse', 'raceway', 'raft', 'railroad_track', 'rainforest', 'reception', 'recreation_room', 'repair_shop', 'residential_neighborhood', 'restaurant', 'restaurant_kitchen', 'restaurant_patio', 'rice_paddy', 'river', 'rock_arch', 'roof_garden', 'rope_bridge', 'ruin', 'runway', 'sandbox', 'sauna', 'schoolhouse', 'science_museum', 'server_room', 'shed', 'shoe_shop', 'shopfront', 'shopping_mall_indoor', 'shower', 'ski_resort', 'ski_slope', 'sky', 'skyscraper', 'slum', 'snowfield', 'soccer_field', 'stable', 'stadium_baseball', 'stadium_football', 'stadium_soccer', 'stage_indoor', 'stage_outdoor', 'staircase', 'storage_room', 'street', 'subway_station_platform', 'supermarket', 'sushi_bar', 'swamp', 'swimming_hole', 'swimming_pool_indoor', 'swimming_pool_outdoor', 'synagogue_outdoor', 'television_room', 'television_studio', 'temple_asia', 'throne_room', 'ticket_booth', 'topiary_garden', 'tower', 'toyshop', 'train_interior', 'train_station_platform', 'tree_farm', 'tree_house', 'trench', 'tundra', 'underwater_ocean_deep', 'utility_room', 'valley', 'vegetable_garden', 'veterinarians_office', 'viaduct', 'village', 'vineyard', 'volcano', 'volleyball_court_outdoor', 'waiting_room', 'water_park', 'water_tower', 'waterfall', 'watering_hole', 'wave', 'wet_bar', 'wheat_field', 'wind_farm', 'windmill', 'yard', 'youth_hostel', 'zen_garden']

def predict_category(img_name):
    with tf.Session() as sess:
        tf.saved_model.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)

        # Get the input and output tensors
        input_tensor = sess.graph.get_tensor_by_name('data:0')
        output_tensor = sess.graph.get_tensor_by_name('prob:0')

        # Load the input image
        image_path =  img_name
        #print(image_path)
        image = cv2.imread(image_path)

        # Resize the image to match the input shape of the model
        input_shape = input_tensor.shape.as_list()[1:3]
        image_resized = cv2.resize(image, tuple(input_shape[::-1]))

        # Expand the image dimensions to create a batch of one image
        image_expanded = image_resized[np.newaxis, ...]

        # Run the prediction
        prediction = sess.run(output_tensor, feed_dict={input_tensor: image_expanded})

        # Get the top 5 predicted class indices and probabilities
        top_k = np.argsort(prediction)[0, ::-1][:5]
        class_indices = top_k
        probabilities = prediction[0, class_indices]

        # Get the class labels for the top 5 predicted class indices
        labels = []
        
        for class_index in class_indices:
            #print(names[class_index])
            labels.append(names[class_index])

        # Return the top 5 class labels and probabilities
        return labels, probabilities
