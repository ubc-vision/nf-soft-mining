document.addEventListener("DOMContentLoaded", function() {
  const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.play();
      } else {
        entry.target.pause();
      }
    });
  }, { threshold: 0.6 }); // Trigger when 50% of the video is visible

  const videos = document.querySelectorAll('video');
  videos.forEach(video => {
    observer.observe(video);
  });
});

document.addEventListener('DOMContentLoaded', () => {
  let carousels = bulmaCarousel.attach('.results-carousel', {
    autoplay: true,
    autoplaySpeed: 10000,
    loop: true,
  });
});