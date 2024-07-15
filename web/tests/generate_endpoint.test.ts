import { describe, it, expect } from 'vitest';

async function* generateImages() {
   const requestBody = {
      models: {
         citron_anime_treasure_v10: 4,
         break_domain_xl_v05g: 1
         // sd3_medium_incl_clips_t5xxlfp8: 1
      },
      positive_prompt:
         'anime style, a badass man swinging a massive broadsword above his head, detailed, epic masterpiece, highly detailed',
      negative_prompt: 'watermark, low quality, worst quality, ugly, text',
      random_seed: 53,
      aspect_ratio: '9/16'
   };

   try {
      const response = await fetch('http://localhost:8881/generate', {
         method: 'POST',
         headers: { 'Content-Type': 'application/json' },
         body: JSON.stringify(requestBody)
      });

      console.log('first received');

      if (!response.ok) {
         throw new Error(`HTTP error! status: ${response.status}`);
      }

      if (!response.body) {
         throw new Error('No body in response');
      }

      const reader = response.body.getReader();

      while (true) {
         const { done, value } = await reader.read();

         if (value) {
            const stringValue = new TextDecoder().decode(value);
            try {
               const jsonValue = JSON.parse(stringValue);

               console.log('message received')
               console.log(`value: ${value}`)

               yield jsonValue;
            } catch (parseError) {
               console.error('Failed to parse JSON:', parseError);
            }
         }

         if (done) break;
      }
   } catch (error) {
      console.error('Failed to generate images:', error);
      throw error;
   }
}

describe('Image Generation API', () => {
  it('should generate images and return valid JSON responses', async () => {
    const chunks: any[] = [];
    
    for await (const chunk of generateImages()) {
      chunks.push(chunk);
      
      // Validate chunk structure
      // expect(chunk).toHaveProperty('image_urls');
      // expect(Array.isArray(chunk.image_urls)).toBe(true);
      
      // chunk.image_urls.forEach((imageUrl: { url: string; is_temp: boolean }) => {
      //   expect(imageUrl).toHaveProperty('url');
      //   expect(typeof imageUrl.url).toBe('string');
      //   expect(imageUrl).toHaveProperty('is_temp');
      //   expect(typeof imageUrl.is_temp).toBe('boolean');
      // });
    }
    
    // Ensure we received at least one chunk
    expect(chunks.length).toBeGreaterThan(0);
  });
}, {
   timeout: 120_000
});

// Invoke generate-images and collect results
// (async () => {
//    try {
//       for await (const chunk of generateImages()) {
//          console.log(chunk);
//       }
//    } catch (error) {
//       console.error('Error in image generation:', error);
//    }
// })();

// fetchData();

//   const response = await fetch(API_Routes.REQUEST_JOB, {
//      method: 'POST',
//      headers: { 'Content-Type': 'application/json', authorization: `Bearer ${idToken}` },
//      body: JSON.stringify(request)
//   });
