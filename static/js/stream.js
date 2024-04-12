const video = document.getElementById("remoteVideo");
const canvas = document.getElementById("canvas");
const context = canvas.getContext("2d");

// Attend que la vidéo soit prête à jouer
// video.addEventListener("canplay", function () {
// });
// Obtenir l'accès à la webcam
if (navigator.mediaDevices.getUserMedia) {
  navigator.mediaDevices
    .getUserMedia({ video: true })
    .then((stream) => (video.srcObject = stream))
    .catch((error) =>
      console.error("Erreur lors de la capture de l'image :", error)
    );
}

// Fonction pour envoyer une image capturée au serveur
function sendImageToServer() {
  const canvas = document.getElementById("canvas");
  const context = canvas.getContext("2d");

  context.drawImage(video, 0, 0, canvas.width, canvas.height); // Assurez-vous que 'video' est l'ID correct

  canvas.toBlob((blob) => {
    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");
    fetch("/process", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.blob())
      .then((blob) => {
        // Traiter la réponse du serveur si nécessaire
        // Convertir le blob en une URL et l'afficher dans un élément <img>
        const imageUrl = URL.createObjectURL(blob);
        document.getElementById("processedImage").src = imageUrl;
      })
      .catch((error) => {
        console.error("Erreur lors de la capture de l'image :", error);
        // document.getElementById("status").textContent = "Erreur: " + error.toString();
      });
  }, "image/jpeg");
}

// Envoyer une image au serveur toutes les secondes
setInterval(sendImageToServer, 10000);
