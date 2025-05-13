document.getElementById("uploadBtn").addEventListener("click", () => {
  const fileInput = document.getElementById("wavInput");
  const file = fileInput.files[0];

  if (!file) {
    displayError("Please select a .wav file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  fetch("http://localhost:5000/files", {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (!response.ok) {
        return response.json().then(err => {
          throw err;
        });
      }
      return response.json();
    })
    .then((data) => {
      saveId(data.id);
      displaySuccess("File uploaded successfully");
      startPolling();
    })
    .catch((error) => {
      console.log(error);
      displayError(error.error);
    });
});

function saveId(id) {
  localStorage.removeItem("id");
  localStorage.setItem("id", id);
}

function getId() {
  return localStorage.getItem("id") || null;
}

let oldid = null;
function displayError(err) {
  if (oldid) {
    clearTimeout(oldid);
  }
  const p = document.getElementById("error");
  p.innerText = err;
  oldid = setTimeout(() => {
    p.innerText = "";
  }, 5000);
}

function displaySuccess(success) {
  if (oldid) {
    clearTimeout(oldid);
  }
  const p = document.getElementById("error");
  p.classList.add("success");
  p.innerText = success;
  oldid = setTimeout(() => {
    p.innerText = "";
    p.classList.remove("success");
  }, 5000);
}


function startPolling() {
  const id = getId();
  if (!id) {
    displayError("No ID found for polling.");
    return;
  }

  const status = document.getElementById("status");
  const downloadBtn = document.getElementById("downloadBtn");
  downloadBtn.disabled = true;
  status.innerText = "⏳ Waiting for your file to be ready...";

  const poll = () => {
    fetch(`http://localhost:5000/files/${encodeURIComponent(id)}`)
      .then(response => {
        if (response.status === 204) {
          console.log("Still waiting... polling again.");
          poll(); // re-issue long poll
          return null;
        }
        if (!response.ok) {
          throw new Error("Polling failed.");
        }
        return response.blob();
      })
      .then(blob => {
        if (!blob) return;

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${id}_synthesised.wav`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);

        downloadBtn.disabled = false;
        downloadBtn.onclick = () => {
          const a2 = document.createElement('a');
          a2.href = url;
          a2.download = `${id}_synthesised.wav`;
          document.body.appendChild(a2);
          a2.click();
          a2.remove();
        };

        status.innerText = "✅ Download completed!";
      })
      .catch(err => {
        console.error(err);
        displayError("Polling error occurred.");
      });
  };

  poll(); // initial call
}
