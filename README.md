# Cuda Burrows Wheeler Transform

 ## Preface
The burrows wheeler transformation is an algorithm used heavily in data compression. It functions as a reversible operation that can be applied to data to make it more compressible. The BWT groups similar characters together, allowing for better compression if Move To Front Encoding in conjunction with Huffman Trees are used on the resulting transform text. This program does not implement Move to Front Encoding, or Huffman Trees. It focuses specifically on the Burrows Wheeler Transform.

## Traditional Implementations
Many applications apply the transform on small blocks of data, of a limited size. The reason for this is the inverse transform is an expensive operation. The complexity of the transform is computationally O(n) if suffix trees are used and spatially O(n). the inverse transformation is a far more expensive operation. From my research the fastest implementation is computationally O(nlog(n)), and Spatially O(n^2), but I did not have the opportunity to research the inverse transformation as much as I had liked to.  

## My Program
Unlike practical implementations, my program applies the transformation to the entire file. It does not break it up into small chunks. This is just to test the theoretical performance of transforming the text. It will correctly, transform and inverse the transformation of any file, binary or text. The transformation can be calculated with the cpu or gpu, however, the inverse transformation was only implemented on the cpu. While large files have been transformed, they have not been reversed because the inverse operation is too expensive to verify. But with smaller files the inverse has been successfully tested, but not larger files due to how long it takes to reverse.
Also, my transformation algorithm was not the most efficient possible. When I started implementing this, I decided to implement in in the way I thought was most efficient. However, after implementing it, I learned much more about suffix arrays, and how they are used to efficiently create the transform. In my implementation I do not directly use suffix arrays. I do use an array of indices to the original text which acts similar to a suffix array, but it is not as efficient. If I could restart this, I would have used suffix arrays to get a true o(n) implementation. In most cases, my algorithm approaches n, but in some cases it is n^2. 

## Usage

    BWT.exe [filepath] [--print | --cpu]
   
 
    BWT.exe input.txt              //transform  
    BWT.exe input.txt.transform    //inverse transform
		
The program will automatically know whether to perform a file transformation or inverse transformation based on the extension. If it ends in .transformed, it will be transformed to the original text and if it doesn't end in that extension it will be transformed to the transform. The optional flag --cpu will use the cpu instead of the gpu. The gpu is the default device for transforming. --print will print the transformation or inverse transformation to the console. If --print is included, it will only be output to a file.  

## Testing Transformation With Cuda Acceleration
A 3.2  MB text file of Lord of the Rings 110 milliseconds

A 151 MB video file only took 11.284 seconds to transform. 

A 201 MB binary executable took 1 hour and 6 minutes to transform. 
