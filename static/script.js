// Main functionality for the LoL RAG Assistant web interface

document.addEventListener('DOMContentLoaded', function() {
    // Handle example links
    const exampleLinks = document.querySelectorAll('.example-link');
    const queryInput = document.getElementById('query');
    
    exampleLinks.forEach(link => {
      link.addEventListener('click', function(e) {
        e.preventDefault();
        const exampleText = this.textContent;
        queryInput.value = exampleText;
        queryInput.focus();
        
        // Smooth scroll to the query input
        queryInput.scrollIntoView({ behavior: 'smooth' });
      });
    });
    
    // Auto-resize textarea as content grows
    queryInput.addEventListener('input', function() {
      this.style.height = 'auto';
      this.style.height = (this.scrollHeight) + 'px';
    });
    
    // Check if query already has content (from a previous submission)
    if (queryInput.value) {
      queryInput.style.height = 'auto';
      queryInput.style.height = (queryInput.scrollHeight) + 'px';
    }
    
    // Add loading state to form submission
    const form = document.querySelector('form');
    const submitButton = form.querySelector('button[type="submit"]');
    
    form.addEventListener('submit', function() {
      // Save the original button text
      const originalText = submitButton.textContent;
      
      // Change button text and disable it
      submitButton.textContent = 'Processing...';
      submitButton.disabled = true;
      
      // Add loading class to form
      form.classList.add('loading');
      
      // Return to original state after submission (browser might cache)
      setTimeout(() => {
        submitButton.textContent = originalText;
        submitButton.disabled = false;
        form.classList.remove('loading');
      }, 10000); // Assume 10 seconds is max processing time
    });
    
    // Add keyboard shortcut (Ctrl+Enter) to submit form
    queryInput.addEventListener('keydown', function(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        form.submit();
      }
    });
    
    // Copy to clipboard functionality for result
    const resultSection = document.querySelector('.result-section');
    if (resultSection) {
      const result = resultSection.querySelector('.result');
      
      // Add copy button
      const copyButton = document.createElement('button');
      copyButton.textContent = 'Copy';
      copyButton.className = 'copy-button';
      copyButton.style.position = 'absolute';
      copyButton.style.top = '25px';
      copyButton.style.right = '25px';
      copyButton.style.padding = '5px 10px';
      copyButton.style.fontSize = '0.8rem';
      
      // Position the container relatively for absolute positioning of button
      resultSection.style.position = 'relative';
      
      resultSection.appendChild(copyButton);
      
      // Add click handler
      copyButton.addEventListener('click', function() {
        // Get text without HTML tags
        const tempElement = document.createElement('div');
        tempElement.innerHTML = result.innerHTML;
        const textToCopy = tempElement.textContent || tempElement.innerText;
        
        // Copy to clipboard
        navigator.clipboard.writeText(textToCopy).then(() => {
          const originalText = copyButton.textContent;
          copyButton.textContent = 'Copied!';
          
          setTimeout(() => {
            copyButton.textContent = originalText;
          }, 2000);
        });
      });
    }
  });