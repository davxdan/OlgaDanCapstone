//SMU BLUE
//PANTONE: PMS 286
//HEX COLOR: #0033A0;
//RGB: (53,76,161)
//CMYK: (100,65,0,3)

//SMU RED
//PANTONE: PMS 186
//HEX COLOR: #C8102E;
//RGB: (204,0,0)
//CMYK: (0,100,85,3)










//
//Iteratively load an array//////////////////////////////
//int[]
//x = new int[100]; // Array to store x-coordinates
//int count = 0; // Positions stored in array
//void setup() {
//  size(100, 100);
//}
//void draw() {
//  x[count] = mouseX; // Assign new x-coordinate to the array
//  count++; // Increment the counter
//  if (count == x.length) { // If the x array is full,
//    x = expand(x); // double the size of x
//    println(x.length); // Write the new size to the console
//  }
//}

//fancy iteration////////////////////////////////////////////
        //for (int ky = -1; ky <= 1; ky++) {
        //  for (int kx = -1; kx <= 1; kx++) {
        //    int pos = (y + ky)*matrixThis.width + (x + kx);
        //    float valR = red(matrixThis.pixels[pos]);
        //    float valG = green(matrixThis.pixels[pos]);
        //    float valB = blue(matrixThis.pixels[pos]);
        //    red += convolutionKernel[ky+1][kx+1] * valR;
        //    green += convolutionKernel[ky+1][kx+1] * valG;
        //    blue += convolutionKernel[ky+1][kx+1] * valB;
        //  }
        //}


//SIMPLE VERTEXES//////////////////////////////////////////////
//size (400,400);
//background(200);
////noFill();
//translate(0, height/2);
//beginShape();
//for(int i=0; i<width;i++) {
//  //vertex(i,cos(i*PI/180)*height/2);
//  point(i,cos(i*PI/180)*height/2);
//}
//endShape();


//Simple Curves///////////////////////////////////////////////
//int[] coords = {
//  40, 40, 80, 60, 100, 100, 60, 120, 50, 150
//};
//void setup() {
//  //size(200, 200);
//  background(255);
//  smooth();
//  noFill();
//  stroke(0);
//  beginShape();
//  curveVertex(40, 40); // the first control point
//  curveVertex(40, 40); // is also the start point of curve
//  curveVertex(80, 60);
//  curveVertex(100, 100);
//  curveVertex(60, 120);
//  curveVertex(50, 150); // the last point of curve
//  curveVertex(50, 150); // is also the last control point
//  endShape();
//  // Use the array to keep the code shorter;
//  // you already know how to draw ellipses!
//  fill(255, 0, 0);
//  noStroke();
//  for (int i = 0; i < coords.length; i += 2) {
//    ellipse(coords[i], coords[i + 1], 3, 3);
//  }
//}


//Example of getting hasmap form JSON
//JSONObject myObject;
//HashMap<String, String> myHashmap;
//// Keys to get from JSON source.
//String[] myKeys = {
//  "preview",
//  "preview_url",
//  "bib_key",
//  "thumbnail_url",
//  "info_url"
//};
//void settings()
//{
//  myHashmap = new HashMap<String, String>();
//  myObject = loadJSONObject("https://openlibrary.org/api/books?bibkeys=ISBN:0385472579&format=json");
//  // Print source JSON
//  println(myObject);
//  JSONObject dataObject = myObject.getJSONObject("ISBN:0385472579");
//  // Fill out the hashmap with the given keys.
//  for(String myKey : myKeys)
//  {
//    myHashmap.put(myKey, dataObject.getString(myKey));
//  }
//  // Print the hashmap
//  for(HashMap.Entry entry : myHashmap.entrySet())
//  {
//    println(entry.getKey() + " : " + entry.getValue());
//  }
//}
