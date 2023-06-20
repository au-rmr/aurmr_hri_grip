import 'photoswipe/dist/photoswipe.css'

import { Gallery, Item } from 'react-photoswipe-gallery'
// import React from 'react';

const ImagePrompt = (props: {images: string[], id: number}) => (
  <Gallery>
    {props.images.map((image, index) => (
        
    <Item
      original={"./item_images/"+String(props.id).padStart(3, '0')+"_"+image+".jpg"}
      thumbnail={"./item_images/"+String(props.id).padStart(3, '0')+"_"+image+".jpg"}
      width="3024"
      height="4032"
    >
      {({ ref, open }) => (
        <img style={{width:'60px', height:'60px'}} ref={ref as React.RefObject<HTMLImageElement>} onClick={open} src={"./item_images/"+String(props.id).padStart(3, '0')+"_"+image+".jpg"} />
      )}
    </Item>
    ))}
  </Gallery>
);

export default ImagePrompt;