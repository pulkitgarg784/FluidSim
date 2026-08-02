#ifndef WHITEWATER_SHARED_GLSL
#define WHITEWATER_SHARED_GLSL

struct WallContact {
  vec3 inward;
  float proximity;
  float clearance;
};

// A particle against a wall does not have neighbors on that side, which would
// otherwise read as spray. The open top of the tank is excluded.
WallContact wallContact(vec3 position) {
  WallContact contact;
  contact.inward = vec3(0.0);
  contact.proximity = 0.0;
  contact.clearance = 1.0;

  vec3 halfBounds = 0.5 * P.boundsSize.xyz;
  float h = P.smoothingRadius;
  for (int axis = 0; axis < 3; ++axis) {
    if (axis == 1 && position.y >= 0.0)
      continue;
    float distance = halfBounds[axis] - abs(position[axis]);
    contact.clearance =
        min(contact.clearance, clamp((distance - 0.5 * h) / h, 0.0, 1.0));
    float proximity = 1.0 - clamp(distance / h, 0.0, 1.0);
    if (proximity > 0.0) {
      contact.inward[axis] = -sign(position[axis]) * proximity;
      contact.proximity += proximity;
    }
  }
  return contact;
}


uint wallNeighbours(WallContact contact, uint neighbourCount) {
  return neighbourCount > 0u
             ? uint(round(min(10.0 * contact.proximity, 16.0)))
             : 0u;
}

vec3 maskWallDirections(vec3 outward, WallContact contact) {
  for (int axis = 0; axis < 3; ++axis) {
    if (contact.inward[axis] != 0.0 &&
        outward[axis] * contact.inward[axis] < 0.0)
      outward[axis] = 0.0;
  }
  return outward;
}

float neighbourBand() {
  return max(float(W.bubbleMinNeighbours - W.sprayMaxNeighbours), 1.0);
}

// Sparse neighbourhoods are airborne spray, dense ones are submerged bubbles,
// and rest is surface foam.
float classify(uint wallAdjustedCount, uint neighbourCount) {
  if (wallAdjustedCount <= W.sprayMaxNeighbours)
    return SPRAY;
  return neighbourCount >= W.bubbleMinNeighbours ? BUBBLE : FOAM;
}

#endif
