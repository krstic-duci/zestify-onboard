// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

export const config = {
  runtime: 'edge',
};

const RAG_API_URL = process.env.RAG_API_URL || 'http://127.0.0.1:8000';

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    // Forward the request to our FastAPI RAG backend
    const response = await fetch(`${RAG_API_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ messages }),
    });

    if (!response.ok) {
      throw new Error(`RAG API responded with status: ${response.status}`);
    }

    const ragResponse = await response.json();

    // Return the response in the format expected by the AI SDK
    return new Response(
      JSON.stringify({
        role: 'assistant',
        content: ragResponse.response,
        sources: ragResponse.sources,
        metadata: ragResponse.metadata,
      }),
      {
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
  } catch (error) {
    console.error('Error calling RAG API:', error);
    
    return new Response(
      JSON.stringify({
        role: 'assistant',
        content: 'Sorry, I encountered an error while processing your request. Please make sure the RAG API server is running.',
        error: error instanceof Error ? error.message : 'Unknown error',
      }),
      {
        status: 500,
        headers: {
          'Content-Type': 'application/json',
        },
      }
    );
  }
}
