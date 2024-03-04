---
title: FAQ
sidebar_position: 3
image: og/docs/more-resources.jpg
# tags: ['FAQ']
---


## General

#### Q: Why would I use Weaviate as my vector database?

<details>
  <summary>Answer</summary>

> Our goal is three-folded. Firstly, we want to make it as easy as possible for others to create their own semantic systems or vector search engines (hence, our APIs are GraphQL based). Secondly, we have a strong focus on the semantic element (the "knowledge" in "vector databases," if you will). Our ultimate goal is to have Weaviate help you manage, index, and "understand" your data so that you can build newer, better, and faster applications. And thirdly, we want you to be able to run it everywhere. This is the reason why Weaviate comes containerized.

</details>

#### Q: What is the difference between Weaviate and for example Elasticsearch?

<details>
  <summary>Answer</summary>

> Other database systems like Elasticsearch rely on inverted indices, which makes search super fast. Weaviate also uses inverted indices to store data and values. But additionally, Weaviate is also a vector-native search database, which means that data is stored as vectors, which enables semantic search. This combination of data storage is unique, and enables fast, filtered and semantic search from end-to-end.

</details>

#### Q: Do you offer Weaviate as a managed service?

<details>
  <summary>Answer</summary>

> Yes, we do - check out [Weaviate Cloud Services](/pricing).

</details>

## Configuration and setup

#### Q: How should I configure the size of my instance?

<details>
  <summary>Answer</summary>

> You can find this in the [architecture section](/developers/weaviate/concepts/resources.md#an-example-calculation) of the docs.

</details>

#### Q: Do I need to know about Docker (Compose) to use Weaviate?

<details>
  <summary>Answer</summary>

> Weaviate uses Docker images as a means to distribute releases and uses Docker Compose to tie a module-rich runtime together. If you are new to those technologies, we recommend reading the [Docker Introduction for Weaviate Users](https://medium.com/semi-technologies/what-weaviate-users-should-know-about-docker-containers-1601c6afa079).

</details>

#### Q: What happens when the Weaviate Docker container restarts? Is my data in the Weaviate database lost?

<details>
  <summary>Answer</summary>

> There are three levels:
> 1. You have no volume configured (the default in our `Docker Compose` files), if the container restarts (e.g. due to a crash, or because of `docker stop/start`) your data is kept
> 2. You have no volume configured (the default in our `Docker Compose` files), if the container is removed (e.g. from `docker compose down` or `docker rm`) your data is gone
> 3. If a volume is configured, your data is persisted regardless of what happens to the container. They can be completely removed or replaced, next time they start up with a volume, all your data will be there

</details>

## Schema and data structure

#### Q: Are there any 'best practices' or guidelines to consider when designing a schema?

*(E.g. if I was looking to perform a semantic search over a the content of a Book would I look to have Chapter and Paragraph represented in the schema etc, would this be preferred over including the entire content of the novel in a single property?)*

<details>
  <summary>Answer</summary>

> As a rule of thumb, the smaller the units, the more accurate the search will be. Two objects of e.g. a sentence would most likely contain more information in their vector embedding than a common vector (which is essentially just the mean of sentences). At the same time more objects leads to a higher import time and (since each vector also makes up some data) more space. (E.g. when using transformers, a single vector is 768xfloat32 = 3KB. This can easily make a difference if you have millions, etc.) of vectors. As a rule of thumb, the more vectors you have the more memory you're going to need.
>
> So, basically, it's a set of tradeoffs. Personally we've had great success with using paragraphs as individual units, as there's little benefit in going even more granular, but it's still much more precise than whole chapters, etc.
>
> You can use cross-references to link e.g. chapters to paragraphs. Note that resolving a cross-references takes a slight performance penalty. Essentially resolving A1->B1 is the same cost as looking up both A1 and B1 indvidually. This cost however, will probably only matter at really large scale.

</details>

#### Q: Should I use references in my schema?

<details>
  <summary>Answer</summary>

> In short: for convenience you can add relations to your data schema, because you need less code and queries to get data. But resolving references in queries takes some of the performance.
>
> 1. If your ultimate goal is performance, references probably don't add any value, as resolving them adds a cost.
> 2. If your goal is represent complex relationships between your data items, they can help a lot. You can resolve references in a single query, so if you have classes with multiple links, it could definitely be helpful to resolve some of those connections in a single query. On the other hand, if you have a single (bi-directional) reference in your data, you could also just denormalize the links (e.g. with an ID field) and resolve them during search.

</details>

#### Q: Is it possible to create one-to-many relationships in the schema?

<details>
  <summary>Answer</summary>

> Yes, it is possible to reference to one or more objects (Class -> one or more Classes) through cross-references. Referring to lists or arrays of primitives, this will be available [soon](https://github.com/weaviate/weaviate/issues/1611).

</details>

#### Q: What is the difference between `text` and `string` and `valueText` and `valueString`?

<details>
  <summary>Answer</summary>

> The `text` and `string` datatypes differ in tokenization behavior. Note that `string` is now deprecated. Read more in [this section](../config-refs/schema/index.md#property-tokenization) on the differences.

</details>

#### Q: Do Weaviate classes have namespaces?

<details>
  <summary>Answer</summary>

Yes. Each class itself acts like namespaces. Additionally, you can use the [multi-tenancy](../concepts/data.md#multi-tenancy) feature to create isolated storage for each tenant. This is especially useful for use cases where one cluster might be used to store data for multiple customers or users.

</details>

#### Q: Are there restrictions on UUID formatting? Do I have to adhere to any standards?

<details>
  <summary>Answer</summary>

> The UUID must be presented as a string matching the [Canonical Textual representation](https://en.wikipedia.org/wiki/Universally_unique_identifier#Format). If you don't specify a UUID, Weaviate will generate a `v4` i.e. a random UUID. If you generate them yourself you could either use random ones or deterministically determine them based on some fields that you have. For this you'll need to use [`v3` or `v5`](https://en.wikipedia.org/wiki/Universally_unique_identifier#Versions_3_and_5_(namespace_name-based)).

</details>

#### Q: If I do not specify a UUID during adding data objects, will Weaviate create one automatically?

<details>
  <summary>Answer</summary>

> Yes, a UUID will be created if not specified.

</details>

#### Q: Can I use Weaviate to create a traditional knowledge graph?

<details>
  <summary>Answer</summary>

> Yes, you can! Weaviate support ontology, RDF-like definitions in its schema, and it runs out of the box. It is scalable, and the GraphQL API will allow you to query through your knowledge graph easily. But now you are here. We like to suggest you really try its semantic features. After all, you are creating a _knowledge_ graph 😉.

</details>

#### Q: Why does Weaviate have a schema and not an ontology?

<details>
  <summary>Answer</summary>

> We use a schema because it focusses on the representation of your data (in our case in the GraphQL API) but you can use a Weaviate schema to express an ontology. One of Weaviate's core features is that it semantically interprets your schema (and with that your ontology) so that you can search for concepts rather than formally defined entities.

</details>

#### Q: What is the difference between a Weaviate data schema, ontologies and taxonomies?

<details>
  <summary>Answer</summary>

> Read about how taxonomies, ontologies and schemas are related to Weaviate in [this blog post](https://medium.com/semi-technologies/taxonomies-ontologies-and-schemas-how-do-they-relate-to-weaviate-9f76739fc695).

</details>

## Text and language processing

#### Q: How to deal with custom terminology?

<details>
  <summary>Answer</summary>

> Sometimes, users work with custom terminology, which often comes in the form of abbreviations or jargon. You can find more information on how to use the endpoint [here](/developers/weaviate/modules/retriever-vectorizer-modules/text2vec-contextionary.md#extending-the-contextionary-v1modulestext2vec-contextionaryextensions)

</details>

#### Q: How can you index data near-realtime without losing semantic meaning?

<details>
  <summary>Answer</summary>

> Every data object [gets its vector representation](../) based on its semantic meaning. In a nutshell, we calculate the vector position of the data object based on the words and concepts used in the data object. The existing model in the contextionary gives already enough context. If you want to get in the nitty-gritty, you can [browse the code here](https://github.com/weaviate/contextionary/tree/master/server), but you can also ask a [specific question on Stackoverflow](https://stackoverflow.com/tags/weaviate/) and tag it with Weaviate.

</details>

#### Q: Why isn't there a text2vec-contextionary in my language?

<details>
  <summary>Answer</summary>

> Because you are probably one of the first that needs one! Ping us [here on GitHub](https://github.com/weaviate/weaviate/issues), and we will make sure in the next iteration it will become available (unless you want it in [Silbo Gomero](https://en.wikipedia.org/wiki/Silbo_Gomero) or another language which is whistled).

</details>

#### Q: How do you deal with words that have multiple meanings?

<details>
  <summary>Answer</summary>

> How can Weaviate interpret that you mean a company, as in business, and not as the division of the army? We do this based on the structure of the schema and the data you add. A schema in Weaviate might contain a company class with the property name and the value Apple. This simple representation (company, name, apple) is already enough to gravitate the vector position of the data object towards businesses or the iPhone. You can read [here](../) how we do this, or you can ask a specific question on [Stackoverflow](https://stackoverflow.com/tags/weaviate/) and tag it with Weaviate.

</details>

#### Q: Is there support to multiple versions of the query/document embedding models to co-exist at a given time? (helps with live experiments of new model versions)

<details>
  <summary>Answer</summary>

> You can create multiple classes in the Weaviate schema, where one class will act like a namespace in Kubernetes or an index in Elasticsearch. So the spaces will be completely independent, this allows space 1 to use completely different embeddings from space 2. The configured vectorizer is always scoped only to a single class. You can also use Weaviate's Cross-Reference features to make a graph-like connection between an object of Class 1 to the corresponding object of Class 2 to make it easy to see the equivalent in the other space.

</details>

## Queries

#### Q: How can I retrieve the total object count in a class?

<details>
  <summary>Answer</summary>

import HowToGetObjectCount from '/_includes/how.to.get.object.count.mdx';

> This `Aggregate` query returns the total object count in a class.

<HowToGetObjectCount/>

</details>

#### Q: How do I get the cosine similarity from Weaviate's certainty?

<details>
  <summary>Answer</summary>

> To obtain the [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) from weaviate's `certainty`, you can do `cosine_sim = 2*certainty - 1`

</details>

#### Q: The quality of my search results change depending on the specified limit. Why? How can I fix this?

<details>
  <summary>Answer</summary>

Weaviate makes use of ANN indices to serve vector searches. An ANN index is an approximate nearest neighbor index. The "approximate" part refers to an explicit recall-query-speed tradeoff. This trade-off is presented in detail in the [ANN benchmarks section](/developers/weaviate/benchmarks/ann.md#results). For example, a 98% recall for a given set of HNSW parameters means that 2% of results will not match the true nearest neighbors. What build parameters lead to what recall depends on the dataset used. The benchmark pages shows 4 different example datasets. Based on the characteristic of each dataset you can pick the one closest to your production load and draw conclusions about the expected recall for the respective build and query-time parameters.

Generally if you need a higher recall than the default parameters provide you with, you can use stronger parameters. This can either be done at build time (`efConstruction`, `maxConnections`) or at query time (`ef`). Roughly speaking, a higher `ef` value at query time means a more thorough search. It will have a slightly higher latency, but also lead to a slightly better recall.

By changing the specified limit, you are implicitly changing the `ef` parameter. This is because the default `ef` value is set to `-1`, indicating that Weaviate should pick the parameter based on the limit. The dynamic `ef` value is controlled using the configuration fields `dynamicEfMin` which acts as a lower boundary, `dynamicEfMax` which acts as an upper boundary and `dynamicEfFactor` which is the factor to derive the target `ef` based on the limit within the lower and upper boundary.

Example: Using the default parameters `ef=-1`, `dynamicEfMin=100`, `dynamicEfMax=500`, `dynamicEfFactor=8`, you will end up with the following `ef` values based on the limit:

* `limit=1`, dynamically calculated: `ef=1*8=8`. This value is below the lower boundary, so `ef` is set to `100`.
* `limit=20`, dynamically calculated: `ef=20*8=160`. This value is within the boundaries, so `ef` is `160`.
* `limit=100`, dynamically calculated: `ef=100*8=800`. This value is above the upper boundary, so `ef` is set to `500`.

If you need a higher search quality for a given limit you can consider the following options:

1. Instead of using a dynamic `ef` value, use a fixed one that provides the desired recall.
1. If your search quality varies a lot depending on the query-time `ef` values, you should also consider choosing stronger build parameters. The [ANN benchmarks section](/developers/weaviate/benchmarks/ann.md#results) present a combination of many different parameter combination for various datasets.

</details>

#### Q: Why did you use GraphQL instead of SPARQL?

<details>
  <summary>Answer</summary>

> For user experience. We want to make it as simple as possible to integrate Weaviate into your stack, and we believe that GraphQL is the answer to this. The community and client libraries around GraphQL are enormous, and you can use almost all of them with Weaviate.

</details>

## Data management

#### Q: What is the best way to iterate through objects? Can I do paginated API calls?

<details>
  <summary>Answer</summary>

> Yes, Weaviate supports cursor-based iteration as well as pagination through a result set.
>
> To iterate through all objects, you can use the `after` operator with both [REST](../api/rest/objects.md#exhaustive-listing-using-a-cursor-after) and [GraphQL](../api/graphql/additional-operators.md#cursor-with-after).
>
> For pagination through a result set, you can use the `offset` and `limit` operators for GraphQL API calls. Take a look at [this page](../api/graphql/filters.md#pagination-with-offset) which describes how to use these operators, including tips on performance and limitations.

</details>

#### Q: What is best practice for updating data?

<details>
  <summary>Answer</summary>

> Here are top 3 best practices for updating data:
> 1. Use the [batch API](../api/rest/batch.md)
> 2. Start with a small-ish batch size e.g. 100 per batch. Adjust up if it is very fast, adjust down if you run into timeouts
> 3. If you have unidirectional relationships (e.g. `Foo -> Bar`.) it's easiest to first import all `Bar` objects, then import all `Foo` objects with the refs already set. If you have more complex relationships, you can also import the objects without references, then use the [`/v1/batch/references API`](../api/rest/batch.md) to set links between classes in arbitrary directions.

</details>

## Modules

#### Q: Can I connect my own module?

<details>
  <summary>Answer</summary>

> [Yes!](/developers/weaviate/modules/other-modules/custom-modules.md)

</details>

#### Q: Can I train my own text2vec-contextionary vectorizer module?

<details>
  <summary>Answer</summary>

> Not at the moment. You can currently use the [available contextionaries](/developers/weaviate/modules/retriever-vectorizer-modules/text2vec-contextionary.md) in a variety of languages and use the transfer learning feature to add custom concepts if needed.

</details>

## Indexes in Weaviate

#### Q: Does Weaviate use Hnswlib?

<details>
  <summary>Answer</summary>

> No
>
> Weaviate uses a custom implementation of HNSW that overcomes certain limitations of [hnswlib](https://github.com/nmslib/hnswlib), such as durability requirements, CRUD support, pre-filtering, etc.
>
> Custom HNSW implementation in Weaviate references:
>
> - [HNSW plugin (GitHub)](https://github.com/weaviate/weaviate/tree/master/adapters/repos/db/vector/hnsw)
> - [vector dot product ASM](https://github.com/weaviate/weaviate/blob/master/adapters/repos/db/vector/hnsw/distancer/asm/dot_amd64.s)
>
> More information:
>
> - [Weaviate, an ANN Database with CRUD support – DB-Engines.com](https://db-engines.com/en/blog_post/87) ⬅️ best resource on the topic
> - [Weaviate's HNSW implementation in the docs](/developers/weaviate/concepts/vector-index.md#hnsw)
>
> _Note I: HNSW is just one implementation in Weaviate, but Weaviate can support multiple indexing algoritmns as outlined [here](/developers/weaviate/concepts/vector-index.md)_

</details>

#### Q: Are all ANN algorithms potential candidates to become an indexation plugin in Weaviate?

<details>
  <summary>Answer</summary>

> No
>
> Some algorithms (e.g., Annoy or ScaNN) are entirely immutable once built, they can neither be changed nor built up incrementally. Instead, they require you to have all of your vectors present, then you build the algorithm once. After a build, you can only query them, but cannot add more elements or change existing elements. Thus, they aren't capable of the CRUD operations we want to support in Weaviate.

</details>

#### Q: Does Weaviate use pre- or post-filtering ANN index search?

<details>
  <summary>Answer</summary>

> Weaviate currently uses pre-filtering exclusively on filtered ANN search.
> See "How does Weaviate's vector and scalar filtering work" for more details.

</details>

#### Q: How does Weaviate's vector and scalar filtering work?

<details>
  <summary>Answer</summary>

> It's a 2-step process:
>
> 1. The inverted index (which is [built at import time](#q-does-weaviate-use-hnswlib)) queries to produce an allowed list of the specified document ids. Then the ANN index is queried with this allow list (the list being one of the reasons for our custom implementation).
> 2. If we encounter a document id which would be a close match, but isn't on the allow list the id is treated as a candidate (i.e. we add it to our list of links to evaluate), but is never added to the result set. Since we only add allowed IDs to the set, we don't exit early, i.e. before the top `k` elements are reached.
>
> For more information on the technical implementations, see [this video](https://www.youtube.com/watch?v=6hdEJdHWXRE).

</details>

#### What is the maximum number of vector dimensions for embeddings?

<details>
  <summary>Answer</summary>

> As the embedding is currently stored using `uint16`, the maximum possible length is currently 65535.

</details>

## Performance

#### Q: What would you say is more important for query speed in Weaviate: More CPU power, or more RAM?

More concretely: If you had to pick between a machine that has 16 GB of RAM and 2 CPUs, or a machine that has 8 GB of RAM and 4 CPUs, which would you pick?

<details>
  <summary>Answer</summary>

> This is a very difficult to answer 100% correctly, because there are several factors in play:
> * **The vector search itself**. This part is CPU-bound, however only with regards to throughput: A single search is single-threaded. Multiple parallel searches can use multiple threads. So if you measure the time of a single request (otherwise idle), it will be the same whether the machine has 1 core or 100. However, if your QPS approach the throughput of a CPU, you'll see massive benefits by adding more Cores
> * **The retrieval of the objects**. Once the vector search part is done, we are essentially left with a list of n IDs which need to be resolved to actual objects. This is IO-bound in general. However, all disk files are memory-mapped. So generally, more mem will allow you to hold more of the disk state in memory. In real life however, it's not that simple. Searches are rarely evenly distributed. So let's pretend that 90% of searches will return just 10% of objects (because these are more popular search results). Then if those 10% of the disk objects are already cached in mem, there's no benefit in adding more memory.
>
> Taking the above in mind: we can carefully say: If throughput is the problem, increase CPU, if response time is the problem increase mem. However, note that the latter only adds value if there are more things that can be cached. If you have enough mem to cache your entire disk state (or at least the parts that are relevant for most queries), additional memory won't add any additional benefit.
> If we are talking about imports on the other hand, they are almost always CPU-bound because of the cost of creating the HNSW index. So, if you can resize between import and query, my recommendation would be roughly prefer CPUs while importing and then gradually replace CPU with memory at query time - until you see no more benefits. (This assumes that there is a separation between importing and querying which might not always be the case in real life).

</details>

#### Q: Data import takes long / is slow, what is causing this and what can I do?

<details>
  <summary>Answer</summary>

> HNSW is super fast at query time, but slower on vectorization. This means that adding and updating data objects costs relatively more time. You could try [asynchronous indexing](../config-refs/schema/vector-index.md#asynchronous-indexing), which separates data ingestion from vectorization.

</details>

#### Q: How can slow queries be optimized?

<details>
  <summary>Answer</summary>

> Queries containing deeply nested references that need to be filtered or resolved can take some time. Read on optimization strategies [here](./performance.md#costs-of-queries-and-operations).

</details>


#### Q: When scalar and vector search are combined, will the scalar filter happen before or after the nearest neighbor (vector) search?

<details>
  <summary>Answer</summary>

> The mixed structured vector searches in Weaviate are pre-filter. There is an inverted index which is queried first to basically form an allow-list, in the HNSW search the allow list is then used to treat non-allowed doc ids only as nodes to follow connections, but not to add to the result set.

</details>

#### Q: Regarding "filtered vector search": Since this is a two-phase pipeline, how big can that list of IDs get? Do you know how that size might affect query performance?

<details>
  <summary>Answer</summary>

> Essentially the list ids uses the internal doc id which is a `uint64` or 8 bytes per ID. The list can grow as long as you have memory available. So for example with 2GB of free memory, it could hold 250M ids, with 20GB it could hold 2.5B ids, etc.
>
> Performance wise there are two things to consider:
> 1. Building the lookup list
> 2. Filtering the results when vector searching
>
> Building the list is a typical inverted index look up, so depending on the operator this is just a single read on == (or a set of range reads, e.g. for >7, we'd read the value rows from 7 to infinity). This process is pretty efficient, similar to how the same thing would happen in a traditional search engine, such as elasticsearch
>
> Performing the filtering during the vector search depends on whether the filter is very restrictive or very loose. In the case you mentioned where a lot of IDs are included, it will be very efficient. Because the equivalent of an unfiltered search would be the one where your ID list contains all possible IDs. So the HNSW index would behave normally. There is however, a small penalty whenever a list is present: We need to check if the current ID is contained an the allow-list. This is essentially a hashmap lookup, so it should be O(1) per object. Nevertheless, there is a slight performance penalty.
>
> Now the other extreme, a very restrictive list, i.e few IDs on the list, actually takes considerably more time. Because the HNSW index will find neighboring IDs, but since they're not contained, they cannot be added as result candidates, meaning that all we can do with them is evaluating their connections, but not the points themselves. In the extreme case of a list that is very, very restrictive, say just 10 objects out of 1B in the worst case the search would become exhaustive if you the filtered ids are very far from the query. In this extreme case, it would actually be much more efficient to just skip the index and do a brute-force indexless vector search on the 10 ids. So, there is a cut-off when a brute-force search becomes more efficient than a heavily-restricted vector search with HNSW. We do not yet have any optimization to discovery such a cut-off point and skip the index, but this should be fairly simple to implement if this ever becomes an actual problem.

</details>

#### Q: My Weaviate setup is using more memory than what I think is reasonable. How can I debug this?

<details>
  <summary>Answer</summary>

> First of all, make sure your import runs with the latest version of Weaviate, since `v1.12.0`/`v1.12.1` fixed an issue where [too much data was written to disk](https://github.com/weaviate/weaviate/issues/1868) which then lead to unreasonable memory consumption after restarts. If this did not fix the issue yet, please see this post on [how to profile the memory usage of a Weaviate setup](https://stackoverflow.com/a/71793178/5322199).

</details>

## Troubleshooting / debugging

#### Q: How can I print a stack trace of Weaviate?

<details>
  <summary>Answer</summary>

You can do this by sending a `SIGQUIT` signal to the process. This will print a stack trace to the console. The logging level and debugging variables can be set with `LOG_LEVEL` and `DEBUG` [environment variables](https://weaviate.io/developers/weaviate/config-refs/env-vars).

Read more on SIGQUIT [here](https://en.wikipedia.org/wiki/Signal_(IPC)#SIGQUIT) and this [StackOverflow answer](https://stackoverflow.com/questions/19094099/how-to-dump-goroutine-stacktraces/35290196#35290196).

</details>

## Miscellaneous

#### Q: Can I request a feature in Weaviate?

<details>
  <summary>Answer</summary>

> Sure (also, feel free to [issue a pull request](https://github.com/weaviate/weaviate/pulls) 😉) you can [add those requests here](https://github.com/weaviate/weaviate/issues). The only thing you need is a GitHub account, and while you're there, make sure to give us a star 😇.

</details>

#### Q: What is Weaviate's consistency model in a distributed setup?

<details>
  <summary>Answer</summary>

> Weaviate is generally modeled to prefer Availability over Consistency (AP over CP). It is designed to deliver low search latencies under high throughput in situations where availability is more business-critical than consistency. If strict serializability is required on your data, we generally recommend storing your data in a different primary data store, use Weaviate as an auxiliary data store, and set up replication between the two. If you do not need serializability and eventual consistency is enough for your use case, Weaviate can be used as a primary datastore.
>
> Weaviate has no notion of transactions, operations always affect exactly a single key, therefore Serializability is not applicable. In a distributed setup (under development) Weaviate's consistency model is eventual consistency. When a cluster is healthy, all changes are replicated to all affected nodes by the time the write is acknowledged by the user. Objects will immediately be present in search results on all nodes after the import request completes. If a search query occurs concurrently with an import operation nodes may not be in sync yet. This means some nodes might already include the newly added or updated objects, while others don't yet. In a healthy cluster, all nodes will have converged by the time the import request has been completed successfully. If a node is temporarily unavailable and rejoins a cluster it may temporarily be out of sync. It will then sync the missed changes from other replica nodes and eventually serve the same data again.

</details>

#### Q: With your aggregations I could not see how to do time buckets, is this possible?

<details>
  <summary>Answer</summary>

> At the moment, we cannot aggregate over timeseries into time buckets yet, but architecturally there's nothing in the way. If there is demand, this seems like a nice feature request, you can submit an [issue here](https://github.com/weaviate/weaviate/issues). (We're a very small company though and the priority is on Horizontal Scaling at the moment.)

</details>

#### Q: How can I run the latest master branch with Docker Compose?

<details>
  <summary>Answer</summary>

> You can run Weaviate with `Docker Compose`, you can build your own container off the [`master`](https://github.com/weaviate/weaviate) branch. Note that this is not an officially released Weaviate version, so this might contain bugs.
>
> ```sh
> git clone https://github.com/weaviate/weaviate.git
> cd weaviate
> docker build --target weaviate -t name-of-your-weaviate-image .
> ```
>
> Then, make a `docker-compose.yml` file with this new image. For example:
>
> ```yml
> version: '3.4'
> services:
>   weaviate:
>     image: name-of-your-weaviate-image
>     ports:
>       - 8080:8080
>     environment:
>       CONTEXTIONARY_URL: contextionary:9999
>       QUERY_DEFAULTS_LIMIT: 25
>       AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
>       PERSISTENCE_DATA_PATH: './data'
>       ENABLE_MODULES: 'text2vec-contextionary'
>       DEFAULT_VECTORIZER_MODULE: 'text2vec-contextionary'
>       AUTOSCHEMA_ENABLED: 'false'
>   contextionary:
>     environment:
>       OCCURRENCE_WEIGHT_LINEAR_FACTOR: 0.75
>       EXTENSIONS_STORAGE_MODE: weaviate
>       EXTENSIONS_STORAGE_ORIGIN: http://weaviate:8080
>       NEIGHBOR_OCCURRENCE_IGNORE_PERCENTILE: 5
>       ENABLE_COMPOUND_SPLITTING: 'false'
>     image: semitechnologies/contextionary:en0.16.0-v1.0.2
> ```
>
> After the build is complete, you can run this Weaviate build with docker compose:

```bash
docker compose up
```

</details>

## More questions?

import DocsMoreResources from '/_includes/more-resources-docs.md';

<DocsMoreResources />
