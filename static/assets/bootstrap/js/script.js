$(document).ready(function() {
  $('#companyList a').on('click', function(e) {
    e.preventDefault();

    // Get company identifier
    const companyName = $(this).data('company');

    // Fetch and display the details of the selected company
    // Placeholder: Replace with actual fetching logic
    $('#companyTitle').text(`Company Details: ${companyName}`);
    $('#companyDetails').html(`<p>Details about ${companyName}...</p>`);
  });
});
