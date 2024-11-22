package org.tensorflow.plants.detection;


import static org.tensorflow.plants.detection.Constants.Constants.MODEL_PLANTS_TFLITE;
import static org.tensorflow.plants.detection.Constants.Constants.RESULT_DISEASED;
import static org.tensorflow.plants.detection.Constants.Constants.RESULT_HEALTHY;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_PICK_IMAGE = 2;
    private ImageView selectedImageView;
    private TextView resultTextView;
    private Interpreter tflite;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //Inicializo las vistas de la interfaz de usuario
        resultTextView = findViewById(R.id.resultTextView);
        selectedImageView = findViewById(R.id.selectedImageView);

        //Configuración de imageView con las imagenes del modelo importadas en /drawable
        //He escogido 2 imágenes del dataset que son diseased y 3 imágenes del dataset que son healthy
        ImageView imageView1 = findViewById(R.id.imageView1);
        ImageView imageView2 = findViewById(R.id.imageView2);
        ImageView imageView3 = findViewById(R.id.imageView3);
        ImageView imageView4 = findViewById(R.id.imageView4);
        ImageView imageView5 = findViewById(R.id.imageView5);

        //Configuración del click realizado en cada una de las imágenes
        View.OnClickListener imageClickListener = new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                ImageView imageView = (ImageView) view;
                //Actualización de la vista central con la imagen seleccionada
                selectedImageView.setImageDrawable(imageView.getDrawable());
                //Llamada a classifyImage para clasificar la imagen en base al modelo entrenado y mostramos el resultado
                String result = classifyImage(imageView);
                resultTextView.setText(result);
            }
        };
        //Asigno el evento click a cada una de las imágenes.
        imageView1.setOnClickListener(imageClickListener);
        imageView2.setOnClickListener(imageClickListener);
        imageView3.setOnClickListener(imageClickListener);
        imageView4.setOnClickListener(imageClickListener);
        imageView5.setOnClickListener(imageClickListener);

        // Carga del modelo TFLite
        try {
            tflite = new Interpreter(loadModelFile());
        } catch (IOException e) {
            Log.e("MainActivity","Error al cargar el modelo .tflite",e);
        }
    }

    /**
     * Metodo para realizar la carga del modelo .tflite
     */
    private MappedByteBuffer loadModelFile() throws IOException {
        //Abrimos un archivo usando el descriptor del modelo que está en la carpeta assets
        try (FileInputStream is = new FileInputStream(getAssets().openFd(MODEL_PLANTS_TFLITE).getFileDescriptor())) {
            //Definición del canal para leer el fichero
            FileChannel fileChannel = is.getChannel();
            long startOffset = getAssets().openFd(MODEL_PLANTS_TFLITE).getStartOffset();
            long declaredLength = getAssets().openFd(MODEL_PLANTS_TFLITE).getDeclaredLength();
            // Cargamos el archivo en memoria para que sea más rápido usarlo
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    /**
     * Metodo para realizar la clasificación de las imágenes usando el modelo
     */
    private String classifyImage(ImageView imageView) {
        //1. Obtenemos el bitmap de la imagen seleccionada
        Bitmap bitmap = ((BitmapDrawable) imageView.getDrawable()).getBitmap();
        return classifyImageBitmap(bitmap);
    }

    @NonNull
    private String classifyImageBitmap(Bitmap bitmap) {
        //2. Redimensionamos el bitmap de la imagen a 224x224 pixeles
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true);

        //3. Creación de ByteBuffer para almacenar los valores de los píxeles
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(4 * 224 * 224 * 3);
        inputBuffer.order(ByteOrder.nativeOrder());

        //4. Convertimos los bitmap de los píxeles a valores float normalizados entre [0-1]
        int[] intValues = new int[224 * 224];
        resizedBitmap.getPixels(intValues, 0, resizedBitmap.getWidth(), 0, 0, resizedBitmap.getWidth(), resizedBitmap.getHeight());

        for (int val : intValues) {
            //5. Extraemos cada uno de los comopnentes RGB y normalizamos los valores a 0 min y 1 max dividiendo entre 255
            inputBuffer.putFloat(((val >> 16) & 0xFF) / 255.0f);
            inputBuffer.putFloat(((val >> 8) & 0xFF) / 255.0f);
            inputBuffer.putFloat((val & 0xFF) / 255.0f);
        }
        //6. Creación de array para almacenar la predicción
        float[][] output = new float[1][1];
        //7. Ejecutamos el modelo .tflite con los datos de entrada y almacenamos la predicción en la salida output
        tflite.run(inputBuffer, output);

        //Si la salida es mayor que 0.5 la predicción será Diseased y sino será Healthy
        return output[0][0] > 0.5 ? RESULT_DISEASED : RESULT_HEALTHY;
    }

    /*
    * Metodo encargado de gestionar el click en el boton de Galería
    * */
    public void clickGallery(View view) {
        //1. Creamos Intent para abrir la galeria y seleccionar una imagen
        Intent intent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, REQUEST_PICK_IMAGE);
    }

    /*
    *Metodo que se ejecuta cuando vuelve un resultado de otra actividad (en nuestro caso, al seleccionar una foto de la galeria)
    * */
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        //1. Se comprueba que el resultado de la activity sea OK (-1) y que exista algún dato
        if (resultCode == RESULT_OK && data != null) {
            Bitmap bitmap = null;
            // 2. Si el usuario seleccionó una imagen de la galería
            if (requestCode == REQUEST_PICK_IMAGE) {
                Uri selectedImage = data.getData();
                try {
                    // Abrimos el archivo de la imagen desde la galería
                    InputStream imageStream = getContentResolver().openInputStream(Objects.requireNonNull(selectedImage));
                    // Decodificamos el archivo en un Bitmap (formato que Android puede usar para imágenes)
                    bitmap = BitmapFactory.decodeStream(imageStream);
                } catch (FileNotFoundException e) {
                   Log.e("MainActivity","Error onActivityResult"+e);
                }
            }

            if (bitmap != null) {
                //1. Mostramos la imagen en un ImageView definido
                selectedImageView.setImageBitmap(bitmap);
                //2. Clasificamos la imagen utilizando la función classifyImageBitmap que usa el modelo
                String result = classifyImageBitmap(bitmap);
                //3. Se muestra el resultado
                resultTextView.setText(result);
                Log.d("ImageClassifier", result);
            }
        }
    }

}
