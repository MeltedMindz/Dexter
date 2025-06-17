export async function GET() {
  const stream = new ReadableStream({
    start(controller) {
      // Set up SSE headers
      const headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
      };

      // Connect to the VPS SSE stream
      const connectToVPS = async () => {
        try {
          const response = await fetch('http://5.78.71.231:3002/logs', {
            method: 'GET',
            headers: {
              'Accept': 'text/event-stream',
              'Cache-Control': 'no-cache'
            }
          });

          if (!response.ok) {
            throw new Error(`Failed to connect to VPS: ${response.status}`);
          }

          const reader = response.body?.getReader();
          if (!reader) {
            throw new Error('Failed to get response reader');
          }

          // Send initial connection message
          const encoder = new TextEncoder();
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            type: 'connection',
            message: 'Connected to Dexter AI agent log stream via proxy',
            timestamp: new Date().toISOString()
          })}\n\n`));

          // Stream data from VPS to client
          const pump = async () => {
            try {
              while (true) {
                const { done, value } = await reader.read();
                
                if (done) {
                  break;
                }
                
                // Forward the data to the client
                controller.enqueue(value);
              }
            } catch (error) {
              console.error('Error in pump:', error);
              // Send error message to client
              controller.enqueue(encoder.encode(`data: ${JSON.stringify({
                type: 'error',
                message: 'Connection to agent lost, attempting reconnect...',
                timestamp: new Date().toISOString()
              })}\n\n`));
              
              // Attempt to reconnect after delay
              setTimeout(connectToVPS, 5000);
            }
          };

          pump();

        } catch (error) {
          console.error('Error connecting to VPS:', error);
          const encoder = new TextEncoder();
          const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
          controller.enqueue(encoder.encode(`data: ${JSON.stringify({
            type: 'error',
            message: `Failed to connect to agent: ${errorMessage}`,
            timestamp: new Date().toISOString()
          })}\n\n`));
          
          // Retry connection after delay
          setTimeout(connectToVPS, 5000);
        }
      };

      // Start the connection
      connectToVPS();

      // Send periodic heartbeat
      const heartbeat = setInterval(() => {
        const encoder = new TextEncoder();
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({
          type: 'heartbeat',
          timestamp: new Date().toISOString()
        })}\n\n`));
      }, 30000);

      // Cleanup function
      return () => {
        clearInterval(heartbeat);
        controller.close();
      };
    }
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET',
      'Access-Control-Allow-Headers': 'Content-Type',
    }
  });
}