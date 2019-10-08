package MnistNNet
import java.io.{BufferedInputStream, FileInputStream}
import java.util.zip.GZIPInputStream
import jeigen.Shortcuts._
import jeigen._
import jeigen.DenseMatrix
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.api.ndarray.INDArray
import scala.util.Random

/*
object MnistLoader {

  private def gzipInputStream(s: String) = new GZIPInputStream(new BufferedInputStream(new FileInputStream(s)))

  private def read32BitInt(i: GZIPInputStream) = i.read() * 16777216 /*2^24*/ + i.read() * 65536 /*2&16*/ + i.read() * 256 /*2^8*/ + i.read()

  /**
    *
    * @param baseDirectory the directory for the standard mnist images, file names are assumed
    */
  def getMnistImageData(baseDirectory: String): (IndexedSeq[Int], IndexedSeq[Int], IndexedSeq[INDArray], IndexedSeq[INDArray]) = {
    val testLabels = readLabels(s"$baseDirectory/t10k-labels-idx1-ubyte.gz")
    val trainingLabels = readLabels(s"$baseDirectory/train-labels-idx1-ubyte.gz")
    val testImages = readImages(s"$baseDirectory/t10k-images-idx3-ubyte.gz")
    val trainingImages = readImages(s"$baseDirectory/train-images-idx3-ubyte.gz")
    (testLabels, trainingLabels, testImages, trainingImages)
  }

  /**
    *
    * @param filepath the full file path the labels file
    * @return
    */
  def readLabels(filepath: String) = {
    val g = gzipInputStream(filepath)
    val magicNumber = read32BitInt(g) //currently not used for anything, as assumptions are made
    val numberOfLabels = read32BitInt(g)
    1.to(numberOfLabels).map(_ => g.read())
  }

  /**
    *
    * @param filepath the full file path of the images file
    * @return
    */
  def readImages(filepath: String) = {
    val g = gzipInputStream(filepath)
    val magicNumber = read32BitInt(g) //currently not used for anything, as assumptions are made
    val numberOfImages = read32BitInt(g)
    val imageSize = read32BitInt(g) * read32BitInt(g) //cols * rows
    1.to(numberOfImages).map(_ => Nd4j.create(1.to(imageSize).map(_ => g.read().toFloat).toArray))
  }

}

*/

object MnistLoader {

  private def gzipInputStream(s: String) = new GZIPInputStream(new BufferedInputStream(new FileInputStream(s)))

  private def read32BitInt(i: GZIPInputStream) = i.read() * 16777216 /*2^24*/ + i.read() * 65536 /*2&16*/ + i.read() * 256 /*2^8*/ + i.read()

  /**
    *
    * @param baseDirectory the directory for the standard mnist images, file names are assumed
    */
  def getMnistImageData(baseDirectory: String): (Array[Array[Double]], Array[Array[Double]], Array[Array[Double]], Array[Array[Double]]) = {
    val testLabels = readLabels(s"$baseDirectory/t10k-labels-idx1-ubyte.gz")
    val trainingLabels = readLabelsMat(s"$baseDirectory/train-labels-idx1-ubyte.gz")
    val testImages = readImages(s"$baseDirectory/t10k-images-idx3-ubyte.gz")
    val trainingImages = readImages(s"$baseDirectory/train-images-idx3-ubyte.gz")
    (testLabels, trainingLabels, testImages, trainingImages)
  }

  /**
    *
    * @param filepath the full file path the labels file
    * @return
    */
  def readLabels(filepath: String) = {
    val g = gzipInputStream(filepath)
    val magicNumber = read32BitInt(g) //currently not used for anything, as assumptions are made
    val numberOfLabels = read32BitInt(g)
    val lblMatrix:Array[Array[Double]] = Array.ofDim[Double](numberOfLabels,1)
    for (row <- 0 to numberOfLabels - 1)
      lblMatrix(row)(0) = g.read().toDouble
    (lblMatrix)
  }

  def readLabelsMat(filepath: String) = {
    val g = gzipInputStream(filepath)
    val magicNumber = read32BitInt(g) //currently not used for anything, as assumptions are made
    val numberOfLabels = read32BitInt(g)
    val lblMatrix:Array[Array[Double]] = Array.ofDim[Double](numberOfLabels,10)
    for (row <- 0 to numberOfLabels - 1)
      lblMatrix(row)(g.read().toInt) = 1.0D
    (lblMatrix)
  }

  /**
    *
    * @param filepath the full file path of the images file
    * @return
    */
  def readImages(filepath: String) = {
    val g = gzipInputStream(filepath)
    val magicNumber = read32BitInt(g) //currently not used for anything, as assumptions are made
    val numberOfImages = read32BitInt(g)
    val imageSize = read32BitInt(g) * read32BitInt(g) //cols * rows
    val dataMatrix:Array[Array[Double]] = Array.ofDim[Double](numberOfImages,imageSize)
    for (row <- 0 to numberOfImages - 1) {
      for (col <- 0 to imageSize - 1) {
        dataMatrix(row)(col) = g.read().toDouble/256
      }
    }
    (dataMatrix)
  }
  def shuffle(data:Array[Array[Double]],label: Array[Array[Double]]):(Array[Array[Double]],Array[Array[Double]])={
     val rand = new scala.util.Random()
     for ( i <- data.length - 1 to 0 by -1 ){
       val j =  rand.nextInt(i+1)
       val x = data(i)
       data(i) = data(j)
       data(j) = x
       val l = label(i)
       label(i) = label(j)
       label(j) = l
     }
     (data,label)

  }

  def slowshuffle(data:Array[Array[Double]],label: Array[Array[Double]]):(Array[Array[Double]],Array[Array[Double]])={

    val indices = Random.shuffle(0 to data.length - 1).toArray
    (indices collect data, indices collect label)
  }


}
