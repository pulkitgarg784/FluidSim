#include "cs488.h"
CS488Window CS488;

// draw something in each frame
static void draw() {
  // for (int j = 0; j < globalHeight; j++) {
  //     for (int i = 0; i < globalWidth; i++) {
  //         // Scrolling uv texture
  //         FrameBuffer.pixel(i, j) = float3(
  //             fmod((float) i / globalWidth + globalFrameCount * 0.01f, 1.0f),
  //             fmod((float) j / globalHeight + globalFrameCount *
  //             0.01f, 1.0f), fmod(globalFrameCount * 0.01f, 1.0f));
  //     }
  // }

  int maxIterations = linalg::min(globalFrameCount, 10000);
  float aspect = (float)globalHeight / globalWidth;

  float minX = -2.5f;
  float maxX = 1.0f;

  float centerY = 0.0f;
  float yRange = (maxX - minX) * aspect;

  float minY = centerY - yRange * 0.5f;
  float maxY = centerY + yRange * 0.5f;

  for (int j = 0; j < globalHeight; j++) {
    for (int i = 0; i < globalWidth; i++) {
      float x0 = minX + float(i) / globalWidth * (maxX - minX);
      float y0 = minY + float(j) / globalHeight * (maxY - minY);
      float x = 0.0f;
      float y = 0.0f;
      int iteration = 0;
      while (x * x + y * y <= 4.0f && iteration < maxIterations) {
        float x2 = x * x;
        float y2 = y * y;
        float newY = 2.0f * x * y + y0;
        float newX = x2 - y2 + x0;
        x = newX;
        y = newY;
        iteration++;
      }

      // float t = (float) iteration / maxIterations;
      // float3 color(
      //     0.5f + 0.5f * cos(6.2831f * (t + 0.00f)),
      //     0.5f + 0.5f * cos(6.2831f * (t + 0.33f)),
      //     0.5f + 0.5f * cos(6.2831f * (t + 0.67f))
      // );

      float3 color;
      // if ( iteration == maxIterations ) {
      //     color = float3(0,0,0);
      // }
      // else {
      //     color = float3(t, t, t);
      // }

      // Inside Mandelbrot set
      if (iteration >= maxIterations) {
        color = float3(0.0f, 0.0f, 0.0f);
      } else {
        // Smooth iteration count
        float log_zn = log(x * x + y * y) / 2.0f;
        float nu = log(log_zn / log(2.0f)) / log(2.0f);

        float smoothIteration = iteration + 1.0f - nu;

        // Normalize to [0,1]
        float t = smoothIteration / maxIterations;

        // // Nice cosine palette
        // float r = 0.5f + 0.5f * cos(6.28318f * (t + 0.0f));
        // float g = 0.5f + 0.5f * cos(6.28318f * (t + 0.33f));
        // float b = 0.5f + 0.5f * cos(6.28318f * (t + 0.67f));

        float r = 9.0f * (1 - t) * t * t * t;
        float g = 15.0f * (1 - t) * (1 - t) * t * t;
        float b = 8.5f * (1 - t) * (1 - t) * (1 - t) * t;

        color = float3(r, g, b);
      }
      FrameBuffer.pixel(i, j) = color;
    }
  }
}
static void A0(int argc, const char *argv[]) {
  // set the function to be called in the main loop
  CS488.process = draw;
}

// setting up lighting
static PointLightSource light;
static void setupLightSource() {
  // light.position = float3(0.5f, 4.0f, 1.0f); // use this for sponza.obj
  light.position = float3(3.0f, 3.0f, 3.0f);
  light.wattage = float3(1000.0f, 1000.0f, 1000.0f);
  globalScene.addLight(&light);
}

// ======== you probably don't need to modify below in A1 to A3 ========
// loading .obj file from the command line arguments
static TriangleMesh mesh;
static void setupScene(int argc, const char *argv[]) {
  if (argc > 1) {
    bool objLoadSucceed = mesh.load(argv[1]);
    if (!objLoadSucceed) {
      printf("Invalid .obj file.\n");
      printf("Making a single triangle instead.\n");
      mesh.createSingleTriangle();
    }
  } else {
    printf("Specify .obj file in the command line arguments. Example: "
           "CS488.exe cornellbox.obj\n");
    printf("Making a single triangle instead.\n");
    mesh.createSingleTriangle();
  }
  globalScene.addObject(&mesh);
}
static void A1(int argc, const char *argv[]) {
  setupScene(argc, argv);
  setupLightSource();
  globalRenderType = RENDER_RASTERIZE;
}

static void A2(int argc, const char *argv[]) {
  setupScene(argc, argv);
  setupLightSource();
  globalRenderType = RENDER_RAYTRACE;
}

static void A3(int argc, const char *argv[]) {
  globalEnableParticles = true;
  setupLightSource();
  globalRenderType = RENDER_RASTERIZE;
  if (argc > 1)
    globalParticleSystem.sphereMeshFilePath = argv[1];
  globalParticleSystem.initialize();
}
// ======== you probably don't need to modify above in A1 to A3 ========

int main(int argc, const char *argv[]) {
  // A0(argc, argv);
  A1(argc, argv);
  // A2(argc, argv);
  // A3(argc, argv);

  CS488.start();
}
