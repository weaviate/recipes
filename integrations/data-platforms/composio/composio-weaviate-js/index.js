import dotenv from "dotenv";
import weaviate from "weaviate-client";
import { vectorizer } from "weaviate-client";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { AgentExecutor, createToolCallingAgent } from "langchain/agents";
import { LangchainToolSet } from "composio-core";
import { DynamicStructuredTool } from "@langchain/core/tools";
import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";

dotenv.config();

// Connect to Weaviate Cloud
const client = await weaviate.connectToWeaviateCloud(process.env.WEAVIATE_URL, {
  authCredentials: new weaviate.ApiKey(process.env.WEAVIATE_API_KEY),
  skipInitChecks: true,
  headers: {
    "X-OpenAI-Api-Key": process.env.OPENAI_API_KEY, // Replace with your API key
  },
});

// Delete the collection if it exists
await client.collections.delete("GenAIWikipedia");

// Create new collection
const myCollection = await client.collections.create({
  name: "GenAIWikipedia",
  vectorizers: vectorizer.text2VecOpenAI(),
});
console.log(`Collection ${myCollection.name} created!`);

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 400,
  chunkOverlap: 50,
});

// Load and split the pdf into chunks
const loader = new PDFLoader(
  "./Generative_artificial_intelligence.pdf"
);
const docs = await loader.load();

const splitDocs = await splitter.splitDocuments(docs);

const filteredDocs = splitDocs.map((doc) => {
  return {
    pageContent: doc.pageContent,
    metadata: {
      source: doc.metadata.source,
    },
  };
});

// Batch insert the documents into the collection
const response = await myCollection.data.insertMany(filteredDocs);

const collection = client.collections.get("GenAIWikipedia");

// Initialize Composio's toolset
const toolset = new LangchainToolSet({ apiKey: process.env.COMPOSIO_API_KEY });

// Get the user
const userEntity = await toolset.client.getEntity("default");

// Create the trigger for new emails
await userEntity.setupTrigger("gmail", "gmail_new_gmail_message", {
  interval: 10, // interval to poll for new changes, in minutes
});

// Subscribe to the above trigger
toolset.client.triggers.subscribe(async (data) => {
  try {
    console.log("data received", data);

    // Extract the relevant information from the event
    const from = data.originalPayload.payload.headers[16].value;
    const message = data.originalPayload.snippet;
    const id = data.originalPayload.threadId;

    // Execute the agent
    executeAgent("default", { from, message, id });
  } catch (error) {
    console.log("Error: ", error);
  }
});

async function executeAgent(entityName, { from, message, id }) {
  try {
    // Get the entity from the toolset
    const entity = await toolset.client.getEntity(entityName);

    //Get the action for replying to an email
    const tools = await toolset.getActions(
      { actions: ["gmail_reply_to_thread"] },
      entity.id
    );

    // Custom tool schema
    const searchSchema = z.object({
      query: z
        .string()
        .describe("The query to be searched in the Weaviate collection"),
    });

    // Custom tool to search the Weaviate collection
    const customTools = [
      new DynamicStructuredTool({
        name: "searchCollection",
        description:
          "Searches the Weaviate collection for user query and returns the results",
        schema: searchSchema,
        func: async ({ query }) => {
          const result = await collection.query.hybrid(query, {
            limit: 3,
          });
          let stringifiedResponse = "";
          result.objects.forEach((object, idx) => {
            stringifiedResponse += `Search Result: ${idx + 1}:\n`;

            const pageContent = object.properties.pageContent;
            stringifiedResponse += `Page Content: ${pageContent}\n`;

            stringifiedResponse += "\n";
          });

          return stringifiedResponse;
        },
      }),
      ...tools,
    ];
    // Create prompt to pass to the agent
    const prompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "You are an AI email assistant that can write and reply to emails. You have to use the search_document tool to search the Weaviate collection for the user's query. When the user asks you a question, use the search_document tool to search for the query in the Weaviate collection and then answer the question using the search results. Send the answer back to the user in an email.",
      ],
      ["human", "{input}"],
      ["placeholder", "{agent_scratchpad}"],
    ]);

    // Create an instance of ChatOpenAI
    const llm = new ChatOpenAI({
      model: "gpt-4o-mini",
      apiKey: process.env.OPENAI_API_KEY,
    });

    // Prepare the input and the agent
    const body = `This is the mail you have to respond to: ${message}. It's from ${from} and the threadId is ${id}`;
    const agent = await createToolCallingAgent({ llm, tools: customTools, prompt });

    // Create an instance of the AgentExecutor
    const agentExecutor = new AgentExecutor({
      agent,
      tools: customTools,
      verbose: true,
    });

    // Invoke the agent
    const result = await agentExecutor.invoke({ input: body });
    console.log(result.output);
  } catch (error) {
    console.log("Error: ", error);
  }
}
