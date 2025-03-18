# Image renaming tool

This branch has Flask tool to classify form types for social dynamics.
Valid form types can be added to form_types.txt which is line delimited

Running image_rename_app.py runs a server at 127.0.0.1:5000 which will present a user with images which can be manually classified with the select drop down.
Save and Rename will prepend the form_type to the filename for easier targeting of images and questions in bvqa (-f -q)
The app also allows for images to be rotated in increments of 90 degrees (Save Rotation - doesn't automatically rename)
BLANK (default) should be used when forms are fully blank or when no (useful) information is present - this helps to prune the image set.

Although many form variants exist, the similarity in content should allow for questions to be targeted to a form type fairly succesfully.

## Form Name conventions

Where a form number is obvious (e.g. B102) this can be selected, however, many forms have supplementary pages, thus a continuation sheet of the B102 will be classified as B102_supp, and so on.
Other forms with no apparent form number are simply verbosely named for their content - e.g. Military History Record.
