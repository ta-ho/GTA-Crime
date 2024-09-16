using BitMiracle.LibTiff.Classic;
using System.Drawing;
using System.Drawing.Imaging;

namespace GTACrime
{
    public class ImageUtils
    {
        public static void WriteToTiff(string dataPath, int width, int height, Bitmap color/*(edit), byte[] depth, byte[] stencil*/)
        {
            try
            {
                color.Save(dataPath + ".tiff", ImageFormat.Tiff);
            }
            catch
            {

            }
            /*(edit)
            try
            {
                var depthTiff = Tiff.Open(dataPath + "-depth.tiff", "w");
                depthTiff.CreateDirectory();
                depthTiff.SetField(TiffTag.IMAGEWIDTH, width);
                depthTiff.SetField(TiffTag.IMAGELENGTH, height);
                depthTiff.SetField(TiffTag.ROWSPERSTRIP, height);
                depthTiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                depthTiff.SetField(TiffTag.SAMPLESPERPIXEL, 1);
                depthTiff.SetField(TiffTag.BITSPERSAMPLE, 32);
                depthTiff.SetField(TiffTag.SUBFILETYPE, FileType.PAGE);
                depthTiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
                depthTiff.SetField(TiffTag.COMPRESSION, Compression.LZW);
                depthTiff.SetField(TiffTag.PREDICTOR, Predictor.FLOATINGPOINT);
                depthTiff.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.IEEEFP);
                depthTiff.SetField(TiffTag.PAGENUMBER, 0, 1);
                //(edit) depthTiff.WriteEncodedStrip(0, depth, depth.Length);
                depthTiff.WriteDirectory();
                depthTiff.Flush();
                depthTiff.Close();
            }
            catch
            {

            }
            try
            {
                var stencilTiff = Tiff.Open(dataPath + "-stencil.tiff", "w");
                stencilTiff.CreateDirectory();
                stencilTiff.SetField(TiffTag.IMAGEWIDTH, width);
                stencilTiff.SetField(TiffTag.IMAGELENGTH, height);
                stencilTiff.SetField(TiffTag.ROWSPERSTRIP, height);
                stencilTiff.SetField(TiffTag.PLANARCONFIG, PlanarConfig.CONTIG);
                stencilTiff.SetField(TiffTag.SAMPLESPERPIXEL, 1);
                stencilTiff.SetField(TiffTag.BITSPERSAMPLE, 8);
                stencilTiff.SetField(TiffTag.SUBFILETYPE, FileType.PAGE);
                stencilTiff.SetField(TiffTag.PHOTOMETRIC, Photometric.MINISBLACK);
                stencilTiff.SetField(TiffTag.COMPRESSION, Compression.LZW);
                stencilTiff.SetField(TiffTag.PREDICTOR, Predictor.HORIZONTAL);
                stencilTiff.SetField(TiffTag.SAMPLEFORMAT, SampleFormat.UINT);
                stencilTiff.SetField(TiffTag.PAGENUMBER, 0, 1);
                //(edit) stencilTiff.WriteEncodedStrip(0, stencil, stencil.Length);
                stencilTiff.WriteDirectory();
                stencilTiff.Flush();
                stencilTiff.Close();
            }
            catch
            {

            }
            */
        }

    }
}
