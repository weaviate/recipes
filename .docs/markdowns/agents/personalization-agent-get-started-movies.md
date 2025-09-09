---
layout: recipe
toc: True
title: "Build a Weaviate Personalization Agent - Movie Recommender"
featured: True
integration: False
agent: True
tags: ['Personalization Agent']
---
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/weaviate/recipes/blob/main/weaviate-services/agents/personalization-agent-get-started-movies.ipynb)

# Build a Weaviate Personalization Agent - Movie Recommender

In this recipe, we will use the new Weaviate `PersonalizationAgent` to fetch personalized objects from a Weaviate collection, in a user personalized way. This new agentic way of retrieving objects is based on a users persona profile and past interactions with your collection.

> 📚 You can learn more about how to use the `PersonalizationAgent`, in our ["Introducing the Weaviate Personalization Agent"](https://weaviate.io/blog/personalization-agent?utm_source=recipe&utm_campaign=agents) blog and [documentation](https://weaviate.io/developers/agents/personalization?utm_source=recipe&utm_campaign=agents).

To help you get started, we're providing a few demo datasets, available on Hugging Face datasets 🤗:

- [Movies](https://huggingface.co/datasets/weaviate/agents/viewer/personalization-agent-movies): A dataset that lists movies, their ratings, original language etc.
- [Recipes](https://huggingface.co/datasets/weaviate/agents/viewer/personalization-agent-recipes): A dataset that lists the name, short description and cuisine of a dish.

For this example, we will be using the movies dataset to create a movie recommender service

```python
!pip install weaviate-client[agents] datasets
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

Next, we create a new collection in Weaviate called "Movies". For the agentic services in Weaviate, it's a good idea to include descriptions of the properties in your collection. These descriptions can then be used by the agent.

```python
from weaviate.classes.config import Configure, DataType, Property

# if client.collections.exists("Movies"):
    # client.collections.delete("Movies")

client.collections.create(
    "Movies",
    description="A dataset that lists movies, including their release dates, ratings, popularity etc.",
    vectorizer_config=Configure.Vectorizer.text2vec_weaviate(),
    properties=[
        Property(
            name="release_date",
            data_type=DataType.TEXT,
            description="release date of the movie",
            skip_vectorization=True,
        ),
        Property(
            name="title", data_type=DataType.TEXT, description="title of the movie"
        ),
        Property(
            name="overview",
            data_type=DataType.TEXT,
            description="overview of the movie",
        ),
        Property(
            name="genres",
            data_type=DataType.TEXT_ARRAY,
            description="genres of the movie",
        ),
        Property(
            name="vote_average",
            data_type=DataType.NUMBER,
            description="vote average of the movie",
        ),
        Property(
            name="vote_count",
            data_type=DataType.INT,
            description="vote count of the movie",
        ),
        Property(
            name="popularity",
            data_type=DataType.NUMBER,
            description="popularity of the movie",
        ),
        Property(
            name="poster_url",
            data_type=DataType.TEXT,
            description="poster path of the movie",
            skip_vectorization=True,
        ),
        Property(
            name="original_language",
            data_type=DataType.TEXT,
            description="Code of the language of the movie",
            skip_vectorization=True,
        ),
    ],
)
```

Python output:
```text
<weaviate.collections.collection.sync.Collection at 0x7f62f442b250>
```
```python
from datasets import load_dataset

dataset = load_dataset("weaviate/agents", "personalization-agent-movies", split="train", streaming=True)

movies_collection = client.collections.get("Movies")

with movies_collection.batch.dynamic() as batch:
    for item in dataset:
        batch.add_object(properties=item["properties"])
```

## Create a Personalization Agent

Below, we create a `PersonalizationAgent` for the `"Movies"` collection. If an agent for this collection already exists, we can simply connect to it.

When creating a new `PeresonalizationAgent`, we can also optioanlly define `user_properties`.

User properties can be anything that may be useful iformation about users that will be added to the agent. In this case, since we are creating a Movie recommender service, we may ask each persona to be added with ther `age`, `favorite_genres` and `languages`.

```python
from weaviate.agents.personalization import PersonalizationAgent

if PersonalizationAgent.exists(client, "Movies"):
  agent = PersonalizationAgent.connect(
          client=client,
          reference_collection="Movies",
      )
else:
  agent = PersonalizationAgent.create(
          client=client,
          reference_collection="Movies",
          user_properties={
            "age": DataType.NUMBER,
            "favorite_genres": DataType.TEXT_ARRAY,
            "languages": DataType.TEXT_ARRAY,
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
              "age": 29,
              "favorite_genres": ["RomCom", "Adventure", "Sci-Fi", "Fantasy"],
              "languages": ["English", "French"],
          },
    )
)
```

### Adding Interactions

Once we have at least one persona for our agent, we can start adding interactions for that persona. For example, in this movie recommender service, it makes sense to add a personas movie reviews.

Each interaction can have a weight between -1.0 (negative) and 1.0 positive. So, we can add some reviews for a number or films below.

It's a good idea to think about what kind of end application may be forwarding these interactions and have a rule around what each weight might represent. For example:
- 1.0: favorite movie  
- 0.8: user liked the movie
- 0.5: user viewed and did not review the movie
- -0.5: user disliked the movie
- -1.0: user absolutely hated the movie 👎

```python
from uuid import UUID
from weaviate.collections.classes.filters import Filter

reviewed_movies = [
    "Fantastic Beasts and Where to Find Them",
    "The Emoji Movie",
    "Titanic",
    "The Shining",
    "Jumanji",
    "West Side Story",
    "The Shape of Water",
    "Morbius"
]

reviews_dict = {
    movie.properties["title"]: movie
    for movie in movies_collection.query.fetch_objects(
        filters=Filter.by_property("title").contains_any(reviewed_movies), limit=20
    ).objects
}

interactions = [
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Fantastic Beasts and Where to Find Them"].uuid, weight=0.8
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["The Emoji Movie"].uuid, weight=-0.5
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Titanic"].uuid, weight=0.5
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["The Shining"].uuid, weight=0.5
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Jumanji"].uuid, weight=0.8
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["West Side Story"].uuid, weight=-0.5
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["The Shape of Water"].uuid, weight=-1.0
    ),
    PersonaInteraction(
        persona_id=persona_id, item_id=reviews_dict["Morbius"].uuid, weight=-1.0
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
    print(obj.properties["overview"])
    print(obj.properties["genres"])
    print(f"vote_average: {obj.properties['vote_average']}")
    print(obj.properties['poster_url'])
```

Python output:
```text
We've placed a spotlight on fantasy and adventure titles given your love for these genres. Movies like 'The Chronicles of Narnia' and 'Jumanji' have been prioritized as they align with your past favorites and preferences. Holiday-themed family films were also considered due to their family-friendly and adventurous nature.
*****0*****
The Chronicles of Narnia: The Lion, the Witch and the Wardrobe
Siblings Lucy, Edmund, Susan and Peter step through a magical wardrobe and find the land of Narnia. There, they discover a charming, once peaceful kingdom that has been plunged into eternal winter by the evil White Witch, Jadis. Aided by the wise and magnificent lion, Aslan, the children lead Narnia into a spectacular, climactic battle to be free of the Witch's glacial powers forever.
['Adventure', 'Family', 'Fantasy']
vote_average: 7.1
https://image.tmdb.org/t/p/original/kzJip9vndXYSKQHCgekrgqbnUrA.jpg
*****1*****
The Ewok Adventure
The Towani family civilian shuttlecraft crashes on the forest moon of Endor. The four Towani's are separated. Jermitt and Catarine, the mother and father are captured by the giant Gorax, and Mace and Cindel, the son and daughter, are missing when they are captured. The next day, the Ewok Deej is looking for his two sons when they find Cindel all alone in the shuttle (Mace and Cindel were looking for the transmitter to send a distress call), when Mace appears with his emergency blaster. Eventually, the four-year old Cindel is able to convince the teenage Mace that the Ewoks are nice. Then, the Ewoks and the Towani's go on an adventure to find the elder Towanis.
['Adventure', 'Family', 'Fantasy', 'Science Fiction', 'TV Movie']
vote_average: 6.0
https://image.tmdb.org/t/p/original/lP7FIxojVrgWsam9efElk5ba3I5.jpg
*****2*****
The Chronicles of Narnia: Prince Caspian
One year after their incredible adventures in the Lion, the Witch and the Wardrobe, Peter, Edmund, Lucy and Susan Pevensie return to Narnia to aid a young prince whose life has been threatened by the evil King Miraz. Now, with the help of a colorful cast of new characters, including Trufflehunter the badger and Nikabrik the dwarf, the Pevensie clan embarks on an incredible quest to ensure that Narnia is returned to its rightful heir.
['Adventure', 'Family', 'Fantasy']
vote_average: 6.6
https://image.tmdb.org/t/p/original/qxz3WIyjZiSKUhaTIEJ3c1GcC9z.jpg
*****3*****
Pete's Dragon
For years, old wood carver Mr. Meacham has delighted local children with his tales of the fierce dragon that resides deep in the woods of the Pacific Northwest. To his daughter, Grace, who works as a forest ranger, these stories are little more than tall tales... until she meets Pete, a mysterious 10-year-old with no family and no home who claims to live in the woods with a giant, green dragon named Elliott. And from Pete's descriptions, Elliott seems remarkably similar to the dragon from Mr. Meacham's stories. With the help of Natalie, an 11-year-old girl whose father Jack owns the local lumber mill, Grace sets out to determine where Pete came from, where he belongs, and the truth about this dragon.
['Adventure', 'Family', 'Fantasy']
vote_average: 6.4
https://image.tmdb.org/t/p/original/6TwrPngfpbwtVH6UsDfVNUnn3ms.jpg
*****4*****
The Chronicles of Narnia: The Voyage of the Dawn Treader
This time around Edmund and Lucy Pevensie, along with their pesky cousin Eustace Scrubb find themselves swallowed into a painting and on to a fantastic Narnian ship headed for the very edges of the world.
['Adventure', 'Family', 'Fantasy']
vote_average: 6.4
https://image.tmdb.org/t/p/original/pP27zlm9yeKrCeDZLFLP2HKELot.jpg
*****5*****
One Piece: Clockwork Island Adventure
Relaxing on a cozy beach, the Straw Hat Pirates are taking a rest from their quest. Right until Luffy noticed the Going Merry has been hijacked and sailed off from the beach. This leads them to search the ship and find the thief who took it from them. They ran into a duo named the Theif Brothers, who informed them that their ship was stolen by a group of pirates called the Trump Kyoudai. When they encountered the Trump Pirates, Nami ended up getting kidnapped as well as Luffy's hat. They tracked down the pirates to their base on Clockwork Island. Now Luffy, Zoro, Sanji, Usopp, and the Theif Brothers must reclaim the Going Merry, Save Nami, and get back Shank's straw hat.
['Action', 'Animation', 'Adventure']
vote_average: 6.8
https://image.tmdb.org/t/p/original/mocek0mTd2dX2neCB691iU9le9k.jpg
*****6*****
Good Luck Charlie, It's Christmas!
Teddy Duncan's middle-class family embarks on a road trip from their home in Denver to visit Mrs. Duncans Parents, the Blankenhoopers, in Palm Springs. When they find themselves stranded between Denver and Utah, they try to hitch a ride to Las Vegas with a seemingly normal older couple in a station wagon from Roswell, New Mexico. It turns out that the couple believes they are the victims of alien abduction. The Duncan's must resort to purchasing a clunker Yugo to get to Utah, have their luggage stolen in Las Vegas, and survive a zany Christmas with Grandpa and Grandma Blankenhooper.
['TV Movie', 'Comedy', 'Family']
vote_average: 6.6
https://image.tmdb.org/t/p/original/ecuJMYZM3HQ96mnWtmyXoHb7s7T.jpg
*****7*****
Bedknobs and Broomsticks
Three children evacuated from London during World War II are forced to stay with an eccentric spinster (Eglantine Price). The children's initial fears disappear when they find out she is in fact a trainee witch.
['Adventure', 'Fantasy', 'Comedy', 'Family', 'Music']
vote_average: 7.0
https://image.tmdb.org/t/p/original/3V7UFCu1u8BJLnoCdvdEqPYVaxQ.jpg
*****8*****
Doraemon: Nobita's New Great Adventure Into the Underworld - The Seven Magic Users
Nobita has changed the real world to a world of magic using the "Moshimo Box" in admiration of magic. One day, Nobita meets Chaplain Mangetsu, a magic researcher, and his daughter Miyoko. Nobita learns from them that the Great Satan of the Devildom Star is scheming to invade the Earth. The Chaplain tries to find a way out using ancient books, however, the Devildom Star rapidly approaches the Earth by the minute, causing earthquakes and abnormal weather conditions. Nobita, along with his friends, Doraemon, Shizuka, Suneo, Gaian and Dorami, together with Miyoko, storm into the Devildom Star, just like the "Legendary Seven Heroes" in an ancient book in order to confront the Great Satan and peace returns to the world of magic.
['Family', 'Adventure', 'Animation']
vote_average: 6.9
https://image.tmdb.org/t/p/original/4E0qUqbevK4DAW2RRbvZmjjPsOd.jpg
*****9*****
Barbie: A Perfect Christmas
Join Barbie and her sisters Skipper, Stacie and Chelsea as their holiday vacation plans turn into a most unexpected adventure and heartwarming lesson. After a snowstorm diverts their plane, the girls find themselves far from their New York destination and their holiday dreams. Now stranded at a remote inn in the tiny town of Tannenbaum, the sisters are welcomed by new friends and magical experiences. In appreciation for the wonderful hospitality they receive, they use their musical talents to put on a performance for the whole town. Barbie and her sisters realize the joy of being together is what really makes A Perfect Christmas!
['Animation', 'Family']
vote_average: 6.7
https://image.tmdb.org/t/p/original/u14NrsyD9h505ZXs5Ofm7Tv3AuM.jpg
*****10*****
The Many Adventures of Winnie the Pooh
Whether we’re young or forever young at heart, the Hundred Acre Wood calls to that place in each of us that still believes in magic. Join pals Pooh, Piglet, Kanga, Roo, Owl, Rabbit, Tigger and Christopher Robin as they enjoy their days together and sing their way through adventures.
['Animation', 'Family']
vote_average: 7.2
https://image.tmdb.org/t/p/original/2xwaFVLv5geVrFd81eUttv7OutF.jpg
*****11*****
The Gruffalo's Child
A follow up to the 2009 animated feature and adapted from the childrens' book by Julia Donaldson and Alex Scheffler.  The Gruffalo's child explores the deep dark wood in search of the big bad mouse and meets the Snake, Owl and Fox in the process.  She eventually finds the mouse, who manages to outwit her like the Gruffalo before!
['Adventure', 'Animation', 'Family', 'Fantasy']
vote_average: 7.0
https://image.tmdb.org/t/p/original/n8fu0eATvUWtmR9CsDlL1gwqTcp.jpg
*****12*****
The Tigger Movie
Winnie the Pooh, Piglet, Owl, Kanga, Roo, and Rabbit are preparing a suitable winter home for Eeyore, the perennially dejected donkey, but Tigger's continual bouncing interrupts their efforts. Rabbit suggests that Tigger go find others of his kind to bounce with, but Tigger thinks "the most wonderful thing about tiggers is" he's "the only one!" Just in case though, the joyously jouncy feline sets out to see if he can find relatives.
['Family', 'Animation', 'Comedy']
vote_average: 6.5
https://image.tmdb.org/t/p/original/lxuiGvLHIL1ZyePP7bn6FcKj0Mr.jpg
*****13*****
Return to Never Land
In 1940, the world is besieged by World War II. Wendy, all grown up, has two children; including Jane, who does not believe Wendy's stories about Peter Pan.
['Adventure', 'Fantasy', 'Animation', 'Family']
vote_average: 6.4
https://image.tmdb.org/t/p/original/zfclCksB7vCLA5Rd9HKHMGSz40.jpg
*****14*****
Winnie the Pooh: A Very Merry Pooh Year
It's Christmastime in the Hundred Acre Wood and all of the gang is getting ready with presents and decorations. The gang makes a list of what they want for Christmas and send it to Santa Claus - except that Pooh forgot to ask for something. So he heads out to retrieve the letter and get it to Santa by Christmas...which happens to be tomorrow!
['Animation', 'Family']
vote_average: 6.8
https://image.tmdb.org/t/p/original/1NbobWwoX7kFd1if8aCyCtFRZpu.jpg
*****15*****
Alice in Wonderland
Alice follows a white rabbit down a rabbit-hole into a whimsical Wonderland, where she meets characters like the delightful Cheshire Cat, the clumsy White Knight, a rude caterpillar, and the hot-tempered Queen of Hearts and can grow ten feet tall or shrink to three inches. But will she ever be able to return home?
['Fantasy', 'Family']
vote_average: 6.3
https://image.tmdb.org/t/p/original/kpXmXjqeyuSYCtNuGMytWQLdKX.jpg
*****16*****
Tarzan the Ape Man
James Parker and Harry Holt are on an expedition in Africa in search of the elephant burial grounds that will provide enough ivory to make them rich. Parker's beautiful daughter Jane arrives unexpectedly to join them. Jane is terrified when Tarzan and his ape friends abduct her, but when she returns to her father's expedition she has second thoughts about leaving Tarzan.
['Action', 'Adventure']
vote_average: 6.8
https://image.tmdb.org/t/p/original/sqtdNAktAI3p1iXmaEooaHjMmWd.jpg
*****17*****
Doraemon: Nobita and the Tin Labyrinth
Nobita's dad stumbled upon a strange advertisement of a fantastic resort on television at midnight. Sleepy as he was, he made a reservation even though he didn't even realize he was talking to the advertisement. The next day he discussed with the family their holiday plans, only to realize he could not find the place anywhere on earth. All of a sudden though there was a suitcase in Nobita's room and intrigued as he was, he opened it only to find a portal to a beautiful resort managed by tin robots. Better still, it's absolutely free. It seems that there is a hidden agenda behind the person who invites them there.
['Adventure', 'Animation']
vote_average: 7.3
https://image.tmdb.org/t/p/original/bkIR641RQyqk7hBlk0hui70dkz9.jpg
*****18*****
Casper's Haunted Christmas
Kibosh, supreme ruler of all ghosts, decrees that casper must scare at least one person before Christmas Day so Casper visits Kriss, Massachusetts where he meets the Jollimore family and sets out to complete his mission. As usual, kindhearted Casper has a ghastky time trying to scare anyone; so The Ghostly Trio, fed up with his goody-boo-shoes behavior, secretly hires Casper's look-alike cousin Spooky to do the job-with hilarious results.
['Animation', 'Family', 'Fantasy']
vote_average: 5.4
https://image.tmdb.org/t/p/original/3BFR30kh0O3NKR1Sfea3HXCG6hw.jpg
*****19*****
Home Alone: The Holiday Heist
8-year-old Finn is terrified to learn his family is relocating from sunny California to Maine in the scariest house he has ever seen! Convinced that his new house is haunted, Finn sets up a series of elaborate traps to catch the “ghost” in action. Left home alone with his sister while their parents are stranded across town, Finn’s traps catch a new target – a group of thieves who have targeted Finn’s house.
['Comedy', 'Family', 'TV Movie']
vote_average: 5.3
https://image.tmdb.org/t/p/original/6JPrRC0JPM06y17pUXD6w1xMvKi.jpg
*****20*****
The Three Caballeros
For Donald's birthday he receives a box with three gifts inside. The gifts, a movie projector, a pop-up book, and a pinata, each take Donald on wild adventures through Mexico and South America.
['Animation', 'Family', 'Music']
vote_average: 6.4
https://image.tmdb.org/t/p/original/nMfScRxw9wVLoO7LiEjziFAKLSK.jpg
*****21*****
The Madagascar Penguins in a Christmas Caper
During the holiday season, when the animals of the Central Park Zoo are preparing for Christmas, Private, the youngest of the penguins notices that the Polar Bear is all alone. Assured that nobody should have to spend Christmas alone, Private goes into the city for some last-minute Christmas shopping. Along the way, he gets stuffed into a stocking
['Animation', 'Comedy', 'Family']
vote_average: 6.7
https://image.tmdb.org/t/p/original/gOVdfrRfzQjYwezOxIap13j05d8.jpg
*****22*****
Jumanji: Welcome to the Jungle
The tables are turned as four teenagers are sucked into Jumanji's world - pitted against rhinos, black mambas and an endless variety of jungle traps and puzzles. To survive, they'll play as characters from the game.
['Adventure', 'Action', 'Comedy', 'Fantasy']
vote_average: 6.8
https://image.tmdb.org/t/p/original/pSgXKPU5h6U89ipF7HBYajvYt7j.jpg
*****23*****
Poltergeist
Steve Freeling lives with his wife, Diane, and their three children, Dana, Robbie, and Carol Anne, in Southern California where he sells houses for the company that built the neighborhood. It starts with just a few odd occurrences, such as broken dishes and furniture moving around by itself. However, when he realizes that something truly evil haunts his home, Steve calls in a team of parapsychologists led by Dr. Lesh to help before it's too late.
['Horror']
vote_average: 7.1
https://image.tmdb.org/t/p/original/xPazCcKp62IshnLVf9BLAjf9vgC.jpg
*****24*****
Up and Away
Hodja is a dreamer. He wants to experience the world, but his father insists he stays home and takes over the family's tailor shop. Fortunately, Hodja meets the old rug merchant El Faza, who gives him a flying carpet. In exchange he has to bring the old man's little granddaughter, Diamond, back to Pjort. El Faza can’t travel to the Sultan city himself, as the mighty ruler has imposed a death sentence on El Faza, on the grounds that he has stolen the Sultan's carpet. However, city life isn't quite what Hodja expected, and he only survives because of Emerald, a poor but street smart girl, who teaches him how to manage in the big world. But when Hodja loses his carpet to the power-hungry sultan, his luck seems to run out. Will he complete his mission, find El Faza's granddaughter and return safely back to Pjort?
['Animation', 'Family', 'Comedy']
vote_average: 6.2
https://image.tmdb.org/t/p/original/1WRK69soLEfVFRW1WwE0vWGz1mq.jpg
```
### Get Recommendations with an Instruction

Optionally, you can also provide the agent with an instruction too. This allows the agent LLM to have more context as to what kind of recommendations it could make.

It may also make sense to set a higher limit for the initial ranking, and then filter down to a smaller group after the agent ranking as we've done below 👇

```python
response = agent.get_objects(persona_id,
                             limit=100,
                             use_agent_ranking=True,
                             instruction="""Your task is to recommend a diverse set of movies that the user may
                             like based on their fave genres and past interactions. Try to avoid recommending multiple films from within
                             the same cinematic universe.""",
)

print(response.ranking_rationale)
for i, obj in enumerate(response.objects[:20]):
    print(f"*****{i}*****")
    print(obj.properties["title"])
    print(obj.properties["overview"])
    print(obj.properties["genres"])
    print(f"vote_average: {obj.properties['vote_average']}")
    print(obj.properties['poster_url'])
```

Python output:
```text
We've highlighted a mix of movies from the user's favorite genres — RomCom, Adventure, Sci-Fi, and Fantasy — ensuring a diverse selection. We've also included some lesser-known gems to provide variety while avoiding multiple entries from the same cinematic universe.
*****0*****
Jumanji: Welcome to the Jungle
The tables are turned as four teenagers are sucked into Jumanji's world - pitted against rhinos, black mambas and an endless variety of jungle traps and puzzles. To survive, they'll play as characters from the game.
['Adventure', 'Action', 'Comedy', 'Fantasy']
vote_average: 6.8
https://image.tmdb.org/t/p/original/pSgXKPU5h6U89ipF7HBYajvYt7j.jpg
*****1*****
The Chronicles of Narnia: The Lion, the Witch and the Wardrobe
Siblings Lucy, Edmund, Susan and Peter step through a magical wardrobe and find the land of Narnia. There, they discover a charming, once peaceful kingdom that has been plunged into eternal winter by the evil White Witch, Jadis. Aided by the wise and magnificent lion, Aslan, the children lead Narnia into a spectacular, climactic battle to be free of the Witch's glacial powers forever.
['Adventure', 'Family', 'Fantasy']
vote_average: 7.1
https://image.tmdb.org/t/p/original/kzJip9vndXYSKQHCgekrgqbnUrA.jpg
*****2*****
Alice Through the Looking Glass
Alice Kingsleigh returns to Underland and faces a new adventure in saving the Mad Hatter.
['Adventure', 'Family', 'Fantasy']
vote_average: 6.5
https://image.tmdb.org/t/p/original/kbGamUkYfgKIYIrU8kW5oc0NatZ.jpg
*****3*****
Madagascar
Alex the lion is the king of the urban jungle, the main attraction at New York's Central Park Zoo. He and his best friends—Marty the zebra, Melman the giraffe and Gloria the hippo—have spent their whole lives in blissful captivity before an admiring public and with regular meals provided for them. Not content to leave well enough alone, Marty lets his curiosity get the better of him and makes his escape—with the help of some prodigious penguins—to explore the world.
['Family', 'Animation', 'Adventure', 'Comedy']
vote_average: 6.9
https://image.tmdb.org/t/p/original/uHkmbxb70IQhV4q94MiBe9dqVqv.jpg
*****4*****
Rio
Captured by smugglers when he was just a hatchling, a macaw named Blu never learned to fly and lives a happily domesticated life in Minnesota with his human friend, Linda. Blu is thought to be the last of his kind, but when word comes that Jewel, a lone female, lives in Rio de Janeiro, Blu and Linda go to meet her. Animal smugglers kidnap Blu and Jewel, but the pair soon escape and begin a perilous adventure back to freedom -- and Linda.
['Animation', 'Adventure', 'Comedy', 'Family']
vote_average: 6.7
https://image.tmdb.org/t/p/original/oo7M77GXEyyqDGOhzNNZTzEuDSF.jpg
*****5*****
Alice in Wonderland
Alice, now 19 years old, returns to the whimsical world she first entered as a child and embarks on a journey to discover her true destiny.
['Family', 'Fantasy', 'Adventure']
vote_average: 6.6
https://image.tmdb.org/t/p/original/o0kre9wRCZz3jjSjaru7QU0UtFz.jpg
*****6*****
Peter Pan
Leaving the safety of their nursery behind, Wendy, Michael and John follow Peter Pan to a magical world where childhood lasts forever. But while in Neverland, the kids must face Captain Hook and foil his attempts to get rid of Peter for good.
['Animation', 'Family', 'Adventure', 'Fantasy']
vote_average: 7.2
https://image.tmdb.org/t/p/original/fJJOs1iyrhKfZceANxoPxPwNGF1.jpg
*****7*****
Titanic
101-year-old Rose DeWitt Bukater tells the story of her life aboard the Titanic, 84 years later. A young Rose boards the ship with her mother and fiancé. Meanwhile, Jack Dawson and Fabrizio De Rossi win third-class tickets aboard the ship. Rose tells the whole story from Titanic's departure through to its death—on its first and last voyage—on April 15, 1912.
['Drama', 'Romance']
vote_average: 7.9
https://image.tmdb.org/t/p/original/9xjZS2rlVxm8SFx8kPC3aIGCOYQ.jpg
*****8*****
The Nutcracker and the Four Realms
When Clara’s mother leaves her a mysterious gift, she embarks on a journey to four secret realms—where she discovers her greatest strength could change the world.
['Fantasy', 'Adventure', 'Family']
vote_average: 6.1
https://image.tmdb.org/t/p/original/9vPDY8e7YxLwgVum7YZIUJbr4qc.jpg
*****9*****
Pete's Dragon
For years, old wood carver Mr. Meacham has delighted local children with his tales of the fierce dragon that resides deep in the woods of the Pacific Northwest. To his daughter, Grace, who works as a forest ranger, these stories are little more than tall tales... until she meets Pete, a mysterious 10-year-old with no family and no home who claims to live in the woods with a giant, green dragon named Elliott. And from Pete's descriptions, Elliott seems remarkably similar to the dragon from Mr. Meacham's stories. With the help of Natalie, an 11-year-old girl whose father Jack owns the local lumber mill, Grace sets out to determine where Pete came from, where he belongs, and the truth about this dragon.
['Adventure', 'Family', 'Fantasy']
vote_average: 6.4
https://image.tmdb.org/t/p/original/6TwrPngfpbwtVH6UsDfVNUnn3ms.jpg
*****10*****
The Chronicles of Narnia: Prince Caspian
One year after their incredible adventures in the Lion, the Witch and the Wardrobe, Peter, Edmund, Lucy and Susan Pevensie return to Narnia to aid a young prince whose life has been threatened by the evil King Miraz. Now, with the help of a colorful cast of new characters, including Trufflehunter the badger and Nikabrik the dwarf, the Pevensie clan embarks on an incredible quest to ensure that Narnia is returned to its rightful heir.
['Adventure', 'Family', 'Fantasy']
vote_average: 6.6
https://image.tmdb.org/t/p/original/qxz3WIyjZiSKUhaTIEJ3c1GcC9z.jpg
*****11*****
One Piece: Chopper's Kingdom on the Island of Strange Animals
As the Straw Hat Pirates sail through the Grand Line.A line of geysers erupted from under the Going Merry. And the whole crew find themselves flying over the island. Unfortunatly, Chopper fell off the ship and was separated from his friends. Luffy and the others landed on the other side of the island. Chopper meanwhile finds himself being worshiped as the island's new king by the animals. To make matters worse, a trio of human "horn" hunters are on the island. The leader, Count Butler is a violin playing/horn eating human who wants to eat the island's treasure to inherit immense power. Will Luffy & the rest be able to prevent the count from terrorizing the island? And will they be able to convince Momambi that not all pirates are bad?
['Action', 'Animation', 'Adventure']
vote_average: 6.6
https://image.tmdb.org/t/p/original/8uzFccR8F3h7tC9zfIOT063r91N.jpg
*****12*****
The Ewok Adventure
The Towani family civilian shuttlecraft crashes on the forest moon of Endor. The four Towani's are separated. Jermitt and Catarine, the mother and father are captured by the giant Gorax, and Mace and Cindel, the son and daughter, are missing when they are captured. The next day, the Ewok Deej is looking for his two sons when they find Cindel all alone in the shuttle (Mace and Cindel were looking for the transmitter to send a distress call), when Mace appears with his emergency blaster. Eventually, the four-year old Cindel is able to convince the teenage Mace that the Ewoks are nice. Then, the Ewoks and the Towani's go on an adventure to find the elder Towanis.
['Adventure', 'Family', 'Fantasy', 'Science Fiction', 'TV Movie']
vote_average: 6.0
https://image.tmdb.org/t/p/original/lP7FIxojVrgWsam9efElk5ba3I5.jpg
*****13*****
Doraemon: Nobita and the Tin Labyrinth
Nobita's dad stumbled upon a strange advertisement of a fantastic resort on television at midnight. Sleepy as he was, he made a reservation even though he didn't even realize he was talking to the advertisement. The next day he discussed with the family their holiday plans, only to realize he could not find the place anywhere on earth. All of a sudden though there was a suitcase in Nobita's room and intrigued as he was, he opened it only to find a portal to a beautiful resort managed by tin robots. Better still, it's absolutely free. It seems that there is a hidden agenda behind the person who invites them there.
['Adventure', 'Animation']
vote_average: 7.3
https://image.tmdb.org/t/p/original/bkIR641RQyqk7hBlk0hui70dkz9.jpg
*****14*****
Barbie & Her Sisters in the Great Puppy Adventure
Barbie and her sisters, Skipper, Stacie and Chelsea, and their adorable new puppy friends find unexpected mystery and adventure when they return to their hometown of Willows. While going through mementos in Grandma's attic, the sisters discover an old map, believed to lead to a long-lost treasure buried somewhere in the town. With their puppy pals in tow, the four girls go on an exciting treasure hunt, along the way discovering that the greatest treasure of all is the love and laughter they share as sisters!
['Family', 'Animation', 'Adventure']
vote_average: 7.3
https://image.tmdb.org/t/p/original/3ybjPWweUjPXXqIXFERgPnOQ5O3.jpg
*****15*****
Tinker Bell and the Lost Treasure
A blue harvest moon will rise, allowing the fairies to use a precious moonstone to restore the Pixie Dust Tree, the source of all their magic. But when Tinker Bell accidentally puts all of Pixie Hollow in jeopardy, she must venture out across the sea on a secret quest to set things right.
['Animation', 'Family', 'Adventure', 'Fantasy']
vote_average: 6.7
https://image.tmdb.org/t/p/original/hg1959yuBkHb4BKbIvETQSfxGCT.jpg
*****16*****
Barbie and the Diamond Castle
Liana and Alexa (Barbie and Teresa) are best friends who share everything, including their love of singing. One day while walking through the forest home from the village, the girls meet an old beggar who gives them a magical mirror. As they clean the mirror and sing, a musical apprentice muse named Melody appears in the mirror's surface, and tells the girls about the secret of the Diamond Castle.
['Animation', 'Family']
vote_average: 7.4
https://image.tmdb.org/t/p/original/dvjFM3GgYm3gDZ6Ulw0JurDYs4r.jpg
*****17*****
Doraemon: New Nobita's Great Demon – Peko and the Exploration Party of Five
The film starts with Gian seeing some like spirit and it asks him to accomplish a task. The story starts when a stray dogs are searching some foods in the can. Then a dirty white dog comes out of nowhere. Just then, the dog creates a close stare to the stray dogs and forces to run away. But the dog is disappointed that he doesn't know what to do after searching the garbage, under the pouring rain. After that, Suneo starts saying about the discoverable places on Earth and him and Gian are totally disappointed about that. That then, they are asking favor to Nobita to take them to an undiscovered place to be seen by their naked eyes. Nobita refuses but tells he will try his best. He comes to upstairs to explain Doraemon what the problem is now.
['Family', 'Animation', 'Adventure']
vote_average: 7.0
https://image.tmdb.org/t/p/original/1XxnaUvLuAlfSWEqravAfSiCaFj.jpg
*****18*****
The Wolf and the Lion
After her grandfather's death, 20-year-old Alma decides to go back to her childhood home - a little island in the heart of the majestic Canadian forest. Whilst there, she rescues two helpless cubs: a wolf and a lion. They forge an inseparable bond, but their world soon collapses as the forest ranger discovers the animals and takes them away. The two cub brothers must now embark on a treacherous journey across Canada to be reunited with one another and Alma once more.
['Adventure', 'Family']
vote_average: 7.2
https://image.tmdb.org/t/p/original/aSRvK4kLJORBrVdlFn2wrGx8XPv.jpg
*****19*****
Tinker Bell and the Legend of the NeverBeast
An ancient myth of a massive creature sparks the curiosity of Tinker Bell and her good friend Fawn, an animal fairy who’s not afraid to break the rules to help an animal in need. But this creature is not welcome in Pixie Hollow — and the scout fairies are determined to capture the mysterious beast, who they fear will destroy their home. Fawn must convince her fairy friends to risk everything to rescue the NeverBeast.
['Adventure', 'Animation', 'Family']
vote_average: 7.1
https://image.tmdb.org/t/p/original/bUGX7duQWSm04yAA1rBBfNRe4kY.jpg
```