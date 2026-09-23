const input = document.querySelector('#image-input');
const preview = document.querySelector('#preview');
const copy = document.querySelector('#upload-copy');

if (input && preview) {
  input.addEventListener('change', () => {
    const file = input.files && input.files[0];
    if (!file) return;
    preview.src = URL.createObjectURL(file);
    preview.hidden = false;
    if (copy) copy.hidden = true;
  });
}
