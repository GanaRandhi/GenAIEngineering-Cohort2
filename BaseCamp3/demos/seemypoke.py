import requests
import urllib3
import streamlit as st

urllib3.disable_warnings()

def get_pokemon_info(pokemon_name):
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name.lower()}"
    response = requests.get(url, verify=False)
    print(response.status_code)    
    
    if response.status_code == 200:
        pokemon_data = response.json()
        pokemon_info = {
            "name": pokemon_data["name"],
            "height": pokemon_data["height"],
            "weight": pokemon_data["weight"],
            "abilities": [ability["ability"]["name"] for ability in pokemon_data["abilities"]],
            "types": [type_data["type"]["name"] for type_data in pokemon_data["types"]],
            "sprites": {
                "front_shiny": pokemon_data["sprites"]["front_shiny"]
            }
        }
        return pokemon_info
    elif response.status_code == 443:
        st.write("Unable to connect to the Pokémon API server.")
        return None
    else:
        return None

#pokemon_name = 'Ivysaur'
if __name__ == '__main__':
    st.write("Welcome to the Pokémon Information App!")
    st.write("Enter the name of a Pokémon to get its information:")
    pokemon_name = st.text_input("Pokémon Name")
    poke = get_pokemon_info(pokemon_name)
    st.write(poke)