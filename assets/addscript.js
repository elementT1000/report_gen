//Using the console method to include this in dash application https://ekoopmans.github.io/html2pdf.js/#:~:text=the%20package%20name).-,Console,-If%20you%E2%80%99re%20on
function addScript(url) {
    var script = document.createElement('script');
    script.type = 'application/javascript';
    script.src = url;
    document.head.appendChild(script);
}
addScript('https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.9.3/html2pdf.bundle.min.js');