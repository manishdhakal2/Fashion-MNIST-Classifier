var plusButton=document.getElementById("openfile");
var filee=document.getElementById("fileInput")
plusButton.addEventListener("click",openfile)
filee.addEventListener("change",displayImage)

function openfile(){
    filee.click()
}

function displayImage(event){
    
    const image= event.target.files[0];
    var imgElement=document.createElement('img');
    if(image){
        const imageUrl = URL.createObjectURL(image);
        imgElement.src = imageUrl;
        imgElement.style.width = "16rem"; // Set desired image width
        imgElement.style.height = "auto"
        plusButton.innerHTML="";
        plusButton.appendChild(imgElement);
    }
}