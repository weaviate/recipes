---
layout: recipe
toc: True
title: "Build a Weaviate Personalization Agent - Food Recommender"
featured: False
integration: False
agent: True
tags: ['Personalization Agent']
---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/weaviate-services/agents/personalization-agent-get-started-recipes.ipynb)

# Build a Weaviate Personalization Agent - Food Recommender

In this recipe, we will use the new Weaviate `PersonalizationAgent` to fetch personalized objects from a Weaviate collection, in a user personalized way. This new agentic way of retrieving objects is based on a users persona profile and past interactions with your collection.

> 📚 You can learn more about how to use the `PersonalizationAgent`, in our ["Introducing the Weaviate Personalization Agent"](https://weaviate.io/blog/personalization-agent?utm_source=recipe&utm_campaign=agents) blog and [documentation](https://weaviate.io/developers/agents/personalization?utm_source=recipe&utm_campaign=agents).

To help you get started, we're providing a few demo datasets, available on Hugging Face datasets 🤗:
- [Recipes](https://huggingface.co/datasets/weaviate/agents/viewer/personalization-agent-recipes): A dataset that lists the name, short description and cuisine of a dish.
- [Movies](https://huggingface.co/datasets/weaviate/agents/viewer/personalization-agent-movies): A dataset that lists movies, their ratings, original language etc.

For this example, we will be using the recipes dataset to create a food recommender service

>[Build a Weaviate Personalization Agent - Food Recommender](#scrollTo=RFe3JtuV6G2f)

>>[Setting Up Weaviate & Importing Data](#scrollTo=IIiKq8EmjyTM)

>>>[Create a New Collection](#scrollTo=3-5ycZprDlxz)

>>[Create a Personalization Agent](#scrollTo=gnagQwcrEQ44)

>>>[Adding New Personas](#scrollTo=segYgHwaEcUG)

>>>[Adding Interactions](#scrollTo=8Cs2sRgXFaPA)

>>[Get Recommendations and Rationale](#scrollTo=oGqM48QJGk-7)

>>>[Get Recommendations with an Instruction](#scrollTo=ypmeUzRNGxQE)

```python
!pip install 'weaviate-client[agents]' datasets
```

## Setting Up Weaviate & Importing Data

To use the Weaviate Personalization Agent, first, create a [Weaviate Cloud](tps://weaviate.io/deployment/serverless?utm_source=recipe&utm_campaign=agents) account👇
1. [Create Serverless Weaviate Cloud account](https://weaviate.io/deployment/serverless?utm_source=recipe&utm_campaign=agents) and setup a free [Sandbox](https://weaviate.io/developers/wcs/manage-clusters/create#sandbox-clusters?utm_source=recipe&utm_campaign=agents)
2. Go to 'Embedding' and enable it, by default, this will make it so that we use `Snowflake/snowflake-arctic-embed-l-v2.0` as the embedding model
3. Take note of the `WEAVIATE_URL` and `WEAVIATE_API_KEY` to connect to your cluster below

> Info: We recommend using [Weaviate Embeddings](https://weaviate.io/developers/weaviate/model-providers/weaviate?utm_source=recipe&utm_campaign=agents) so you do not have to provide any extra keys for external embedding providers.

```python
import os

import weaviate
from weaviate.auth import Auth
from getpass import getpass

if "WEAVIATE_API_KEY" not in os.environ:
  os.environ["WEAVIATE_API_KEY"] = getpass("Weaviate API Key")
if "WEAVIATE_URL" not in os.environ:
  os.environ["WEAVIATE_URL"] = getpass("Weaviate URL")

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ.get("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.environ.get("WEAVIATE_API_KEY")),
)
```

### Create a New Collection

Next, we create a new collection in Weaviate called "Recipes". For the agentic services in Weaviate, it's a good idea to include descriptions of the properties in your collection. These descriptions can then be used by the agent.

```python
from weaviate.classes.config import Configure, DataType, Property

# if client.collections.exists("Recipes"):
#     client.collections.delete("Recipes")

client.collections.create(
    "Recipes",
    description="A dataset that lists recipes with titles, desctiptions, and labels indicating cuisine",
    vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
    properties=[
        Property(
            name="title", data_type=DataType.TEXT, description="title of the recipe"
        ),
        Property(
            name="labels",
            data_type=DataType.TEXT,
            description="the cuisine the recipe belongs to",
        ),
        Property(
            name="description",
            data_type=DataType.TEXT,
            description="short description of the recipe",
        ),
    ],
)
```

```python
from datasets import load_dataset

dataset = load_dataset("weaviate/agents", "personalization-agent-recipes", split="train", streaming=True)

recipes_collection = client.collections.get("Recipes")

with recipes_collection.batch.dynamic() as batch:
    for item in dataset:
        batch.add_object(properties=item["properties"])

```

## Create a Personalization Agent

Below, we create a `PersonalizationAgent` for the `"Recipes"` collection. If an agent for this collection already exists, we can simply connect to it.

When creating a new `PeresonalizationAgent`, we can also optioanlly define `user_properties`.

User properties can be anything that may be useful iformation about users that will be added to the agent. In this case, since we are creating a food recommender service, we may ask each persona to be added with ther `favorite_cuisines`, `likes` and `dislikes`.

```python
from weaviate.agents.personalization import PersonalizationAgent

if PersonalizationAgent.exists(client, "Recipes"):
  agent = PersonalizationAgent.connect(
          client=client,
          reference_collection="Recipes",
      )
else:
  agent = PersonalizationAgent.create(
          client=client,
          reference_collection="Recipes",
          user_properties={
              "favorite_cuisines": DataType.TEXT_ARRAY,
              "likes": DataType.TEXT_ARRAY,
              "dislikes": DataType.TEXT_ARRAY
          },
      )

```

### Adding New Personas

We can add new users with `add_persona`, listing the requested user properties when adding them. Try changing the code block below to represent yourself if you like 👇

```python
from uuid import uuid4
from weaviate.agents.classes import Persona, PersonaInteraction

persona_id = uuid4()
agent.add_persona(
    Persona(
        persona_id=persona_id,
        properties={
            "favorite_cuisines": ["Italian", "Thai"],
            "likes": ["chocolate", "salmon", "pasta", "most veggies"],
            "dislikes": ["okra", "mushroom"],
        },
    )
)
```

```python
agent.get_persona(persona_id)

```

Python output:
```text
Persona(persona_id=UUID('df987437-4d10-44d6-b613-dfff31f715fb'), properties={'favorite_cuisines': ['Italian', 'Thai'], 'dislikes': ['okra', 'mushroom'], 'allergies': None, 'likes': ['chocolate', 'salmon', 'pasta', 'most veggies']})
```
### Adding Interactions

Once we have at least one persona for our agent, we can start adding interactions for that persona. For example, in this food recommender service, it makes sense to add a personas food reviews.

Each interaction can have a weight between -1.0 (negative) and 1.0 positive. So, we can add some reviews for a number or dishes below.

It's a good idea to think about what kind of end application may be forwarding these interactions and have a rule around what each weight might represent. For example, let's imagine a recipes website
- 1.0: favorite meal  
- 0.8: user liked the dish
- 0.5: user viewed the recipe page
- -0.5: user disliked the dish
- -1.0: user absolutely hated the dish 👎

```python
from uuid import UUID
from weaviate.collections.classes.filters import Filter

reviewed_foods = [
    "Coq au Vin",
    "Chicken Tikka Masala",
    "Gnocchi alla Sorrentina",
    "Matcha Ice Cream",
    "Fiorentina Steak",
    "Nabe",
    "Duck Confit",
    "Pappardelle with Porcini"
]

reviews_dict = {
    recipe.properties["title"]: recipe
    for recipe in recipes_collection.query.fetch_objects(
        filters=Filter.by_property("title").contains_any(reviewed_foods), limit=20
    ).objects
}

interactions = [
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Coq au Vin"].uuid, weight=0.8
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Chicken Tikka Masala"].uuid, weight=0.8
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Matcha Ice Cream"].uuid, weight=0.8
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Gnocchi alla Sorrentina"].uuid, weight=0.5
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Fiorentina Steak"].uuid, weight=0.8
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Nabe"].uuid, weight=0.5
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Duck Confit"].uuid, weight=1.0
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Pappardelle with Porcini"].uuid, weight=-1.0
    ),

]
```

```python
agent.add_interactions(interactions=interactions)
```

## Get Recommendations and Rationale

Now that we have a persona and some interactions for that persona, we can start getting recommended objects from the agent with `get_objects`. We have two options here: we can set `use_agent_ranking` or not.

When we do not use agent ranking, the returned objects are ranked by classic ML clustering, whereas when we do use it, it will go through an additional re-ranking with an LLM and an optioanl `instruction`.

When we use agent ranking, we can also see the rationale behind the ranking in `ranking_rationale` as we've done below 👇

```python
response = agent.get_objects(persona_id, limit=25, use_agent_ranking=True)

print(response.ranking_rationale)
for i, obj in enumerate(response.objects):
    print(f"*****{i}*****")
    print(obj.properties["title"])
    print(obj.properties["description"])
    print(obj.properties["labels"])
```

Python output:
```text
Based on your love for Italian cuisine and positive interactions with dishes like Gnocchi alla Sorrentina and Fiorentina Steak, Italian dishes like Frittata di Zucca e Pancetta and Classic Italian Margherita Pizza are highlighted. Your fondness for Chicken Tikka Masala also brought Indian dishes such as Spicy Indian Tikka Masala forward. Although you enjoyed Coq au Vin, the included mushrooms might not be to your liking, which is reflected in a balanced way within French dishes.
*****0*****
Frittata di Zucca e Pancetta
A fluffy egg omelette with sweet potatoes and pancetta, seasoned with herbs and grated cheese, a beloved dish from the heart of Italy.
Italian
*****1*****
Pizza Margherita
A simple yet iconic pizza with San Marzano tomatoes, mozzarella di bufala, fresh basil, and extra-virgin olive oil, encapsulating the Neapolitan pizza tradition.
Italian
*****2*****
Lasagna alla Bolognese
Layers of pasta sheets, Bolognese sauce, and béchamel, all baked to golden perfection, embodying the comforting flavors of Emilia-Romagna.
Italian
*****3*****
Lasagna alla Bolognese
Layers of flat pasta sheets, rich Bolognese sauce, and béchamel, baked to perfection.
Italian
*****4*****
Spicy Indian Tikka Masala
A rich tomato and cream sauce with chunks of chicken, covered in a fiery blend of spices and charred chunks of chicken.
Indian
*****5*****
Classic Italian Margherita Pizza
Thin crust pizza topped with San Marzano tomatoes, fresh mozzarella, basil, and extra-virgin olive oil, representing the simplicity of Italian cuisine.
Italian
*****6*****
Chicken Tikka Masala
Marinated chicken drumsticks grilled on a spit and then simmered in a spicy tomato sauce with cream, a popular dish in Indian cuisine.
Indian
*****7*****
Butter Chicken
A creamy and aromatic tomato sauce with tender chunks of chicken, marinated in a blend of spices and cooked with yogurt and cream, often served with rice or naan bread.
Indian
*****8*****
French Coq au Vin
A hearty stew of chicken braised with wine, mushrooms, and garlic, capturing the essence of French country cooking.
French
*****9*****
Sicilian Arancini
Deep-fried balls of risotto mixed with cheese and peas, coated with breadcrumbs and Parmesan cheese.
Italian
*****10*****
Ramen
A noodle soup dish with Chinese influences, typically containing Chinese-style noodles served in a meat or fish-based broth, often flavored with soy sauce or miso, and uses toppings such as sliced pork, green onions, and nori.
Japanese
*****11*****
Oden
A hearty Japanese hotpot dish made with simmered fish cakes, tofu, konnyaku, and vegetables, in a dashi-based broth.
Japanese
*****12*****
Shabu-Shabu
A Japanese hot pot dish where thinly sliced meat and vegetables are cooked in boiling water at the table.
Japanese
*****13*****
Tempura
A Japanese dish usually made with seafood, vegetables, and sometimes meat, battered and deep-fried until crisp.
Japanese
*****14*****
Oden
A Japanese stew containing fish cakes, daikon radish, konnyaku, tofu, and boiled eggs, typically flavored with miso or dashi broth.
Japanese
*****15*****
Tandoori Chicken
Marinated in yogurt and spices, then cooked in a tandoor (clay oven), resulting in a tangy and spicy chicken dish.
Indian
*****16*****
Beef Bourguignon
A flavorful beef stew cooked slowly in red wine, beef broth, and a bouquet garni, with carrots, onions, and mushrooms, typically served with potatoes or noodles.
French
*****17*****
Pizza Margherita
Simple pizza with San Marzano tomatoes, fresh mozzarella, basil, and extra-virgin olive oil, reflecting the colors of the Italian flag.
Italian
*****18*****
Butter Chicken
Soft and tender pieces of chicken in a rich and creamy sauce made with butter, tomatoes, and a blend of Indian spices.
Indian
*****19*****
Chicken Biryani
A flavorful rice dish cooked with basmati rice, chicken, and a mix of aromatic spices such as cardamom, cinnamon, and cloves, layered with meat and vegetables.
Indian
*****20*****
Milanesa a la Napolitana
A breaded cutlet, typically veal, topped with melted cheese and tomato sauce, a popular street food item.
Argentinian
*****21*****
Tempura Udon
Thick udon noodles served with crispy tempura shrimp and vegetables, lightly coated in a dashi-based sauce for a delicate taste.
Japanese
*****22*****
Miso Soup
A traditional Japanese soup consisting of a stock called dashi into which softened miso paste is mixed, often with pieces of tofu and seaweed.
Japanese
*****23*****
Udon Noodles
A type of thick wheat flour noodle common in Japanese cuisine, served hot or cold with a dipping sauce.
Japanese
*****24*****
Pappardelle with Porcini
Thick wide ribbons of pasta served with a creamy porcini mushroom sauce and grated Parmesan cheese.
Italian
```
### Get Recommendations with an Instruction

Optionally, you can also provide the agent with an instruction too. This allows the agent LLM to have more context as to what kind of recommendations it could make.

It may also make sense to set a higher limit for the initial ranking, and then filter down to a smaller group after the agent ranking as we've done below 👇

```python
response = agent.get_objects(persona_id,
                             limit=50,
                             use_agent_ranking=True,
                             instruction="""Your task is to recommend a diverse set of dishes to the user
                             taking into account their likes and dislikes. It's especially important to avoid their dislikes.""",
)

print(response.ranking_rationale)
for i, obj in enumerate(response.objects[:10]):
    print(f"*****{i}*****")
    print(obj.properties["title"])
    print(obj.properties["description"])
    print(obj.properties["labels"])
```

Python output:
```text
As you love Italian cuisine and have a special liking for foods like pasta and salmon, while disliking mushrooms, we've focused on offering you a variety of Italian and other delightful dishes without mushroom content. We've also incorporated a touch of diversity with dishes from other cuisines you enjoy, while carefully avoiding those with ingredients you dislike.
*****0*****
Chicken Tikka Masala
Marinated chicken drumsticks grilled on a spit and then simmered in a spicy tomato sauce with cream, a popular dish in Indian cuisine.
Indian
*****1*****
Pasta alla Norma
Pasta served with fried eggplant, tomato sauce, and ricotta salata, a flavorful dish that showcases the vibrant flavors of Sicilian cuisine.
Italian
*****2*****
Classic Italian Margherita Pizza
Thin crust pizza topped with San Marzano tomatoes, fresh mozzarella, basil, and extra-virgin olive oil, representing the simplicity of Italian cuisine.
Italian
*****3*****
Pizza Margherita
Simple pizza with San Marzano tomatoes, fresh mozzarella, basil, and extra-virgin olive oil, reflecting the colors of the Italian flag.
Italian
*****4*****
Spicy Indian Tikka Masala
A rich tomato and cream sauce with chunks of chicken, covered in a fiery blend of spices and charred chunks of chicken.
Indian
*****5*****
Lasagna alla Bolognese
Layers of flat pasta sheets, rich Bolognese sauce, and béchamel, baked to perfection.
Italian
*****6*****
Fettuccine Alfredo
Creamy pasta dish made with fettuccine pasta tossed in a rich sauce of butter, heavy cream, and Parmesan cheese.
Italian
*****7*****
Sicilian Arancini
Deep-fried balls of risotto mixed with cheese and peas, coated with breadcrumbs and Parmesan cheese.
Italian
*****8*****
Frittata di Zucca e Pancetta
A fluffy egg omelette with sweet potatoes and pancetta, seasoned with herbs and grated cheese, a beloved dish from the heart of Italy.
Italian
*****9*****
Paneer Tikka
Small cubes of paneer marinated in spices and yogurt, then grilled and served in a spicy tomato sauce.
Indian
```