// umap.tsx

// @ts-nocheck // HACK

import React, { useEffect, useState } from "react"

import DeckGL from '@deck.gl/react/typed';
import {BitmapLayer} from '@deck.gl/layers/typed';
import {TileLayer} from '@deck.gl/geo-layers/typed';
import {OrthographicView, COORDINATE_SYSTEM} from '@deck.gl/core/typed';

import GL from '@luma.gl/constants';
import {TextField, List, ListItem} from '@mui/material';

import {
  ComponentProps,
  Streamlit,
  withStreamlitConnection,
} from "streamlit-component-lib"

// --
// Helpers

const _PALLETTE = [
  [0, 0, 0, 0],
  [228, 26, 28, 175],
  [55, 126, 184, 175],
  [77, 175, 74, 175],
  [152, 78, 163, 175],
  [255, 193, 7, 175],
  [166, 86, 40, 175],
  [247, 129, 191, 175],
  [153, 153, 153, 175]
]

const make_colors = function(config, f2label, classes) {  
  const colors = new Uint8Array(config.nrow * config.ncol * 4);
  
  Object.entries(f2label).forEach(([fname, lab]) => {
    const [r, c]  = config.f2coord[fname];
    const offset  = (r * config.ncol + c);
    const lab_idx = classes.indexOf(lab);
    if(lab_idx === -1) return;
    
    colors[4 * offset + 0] = _PALLETTE[lab_idx][0];
    colors[4 * offset + 1] = _PALLETTE[lab_idx][1];
    colors[4 * offset + 2] = _PALLETTE[lab_idx][2];
    colors[4 * offset + 3] = _PALLETTE[lab_idx][3];
  });
  
  return colors;
}

const UMap = (props: ComponentProps) => {
  useEffect(() => Streamlit.setFrameHeight())

  const {config}  = props.args;
  const chip_size = config.tileSize / (2 ** (1 + config.maxZoom));

  const [counter, setCounter]  = useState(0);
  const [classes, setClasses]  = useState([]);
  const [f2label, setF2Label]  = useState({});
  const [curr_lab, setCurrLab] = useState(null);
  const [unsaved, setUnsaved]  = useState(false);

  useEffect(() => { 
    console.log('!!!!!!!!!!!! INIT')
    setF2Label(config.f2label); 
    setClasses(config.classes);
  }, []);
  
  const colors = make_colors(config, f2label, classes)
  
  // --
  // Image Layer
  
  const initialViewState = {  
    main : {
      target : [config.ncol * chip_size / 2, config.nrow * chip_size / 2, 0],
      zoom   : 2
    }
  };

  const umap_layer = new TileLayer({
    id   : `tile-layer-${config.name}`,
    data : config.tiles[0],

    minZoom          : config.minZoom,
    maxZoom          : config.maxZoom,
    tileSize         : config.tileSize,
    coordinateSystem : COORDINATE_SYSTEM.CARTESIAN,
    coordinateOrigin : [0, 0, 0],
    extent           : [0, 0, config.ncol * chip_size, config.nrow * chip_size], // correct?

    onTileError: () => {},
    renderSubLayers: props => {
      const {
        bbox: {left, bottom, right, top}
      } = props.tile;

      return new BitmapLayer(props, {
        data              : null,
        image             : props.data,
        bounds            : [left, bottom, right, top],
        textureParameters : {
          [GL.TEXTURE_MIN_FILTER]: GL.NEAREST,
          [GL.TEXTURE_MAG_FILTER]: GL.NEAREST
        }
      });
    }
  });

  // --
  // Annotation Layer
  
  useEffect(() => {
    
    const _keydown = async (event) => {
      if(parseInt(event.key) <= classes.length) {
        setCurrLab(curr_lab => classes[parseInt(event.key)]);
      } else if(event.key === 'Meta') {
        console.log('!!!!! POST', f2label);
        Streamlit.setComponentValue([f2label, classes]);
        setUnsaved(false);
      } else {
        setCurrLab(curr_lab => null);
      }
    };

    const _keyup = async (event) => {
        setCurrLab(curr_lab => null);
    };

    document.addEventListener('keydown', _keydown);
    document.addEventListener('keyup', _keyup);
    return () => {
      document.removeEventListener('keydown', _keydown);
      document.removeEventListener('keyup', _keyup);
    }
  }, [counter, classes, f2label]);
  
  const ann_layer = new BitmapLayer({
    id    : "ann_layer",
    image : {
      width  : config.ncol,
      height : config.nrow,
      data   : colors,
    },
    textureParameters: {
      [GL.TEXTURE_MIN_FILTER]: GL.NEAREST,
      [GL.TEXTURE_MAG_FILTER]: GL.NEAREST
    },
    updateTriggers : {
      image : [f2label],
    },
    
    // This doesn't seem like it'll always work ... 
    
    extent : [0, 0, config.ncol * chip_size, config.nrow * chip_size], // correct?
    bounds : [0, 0, config.ncol * chip_size, config.nrow * chip_size],
    
    pickable: true,
    
    onHover : ({bitmap}) => {
      if(bitmap) {
        if(curr_lab === null) return;
        
        const [c, r]    = bitmap.pixel; // from bottom left
        const offset    = (r * bitmap.size.width + c);      
        if(!config.fnames[offset]) return;
        
        setCounter(counter + 1);
        setF2Label({
          ...f2label,
          [config.fnames[offset]] : curr_lab
        });
        setUnsaved(true);
      }
    }
  });

  // Map/View
  const main_view = new OrthographicView({
    id         : 'main', 
    controller : {keyboard: {moveSpeed: 100}},
    x          : "0%",
    y          : "0%",
    width      : '100%',
    height     : '100%',    
  });

  return (
    <>
      <div id="left_pane" className="split left">
        <TextField 
          id="class-input" 
          label="Add Classes..." 
          variant="standard"
          style={{paddingRight:"10%"}}
          onKeyDown={(ev) => {
            if (ev.key === 'Enter') {
              if(ev.target.value === "") return;

              if(classes.indexOf(ev.target.value) !== -1) {
                alert(`Alert: ${ev.target.value} already exists`);
                return;
              }

              if(classes.length == 9) {
                alert(`Alert: You've hit the maximum of 9 classes!`);
                return;
              }
              
              ev.preventDefault();
              setClasses([...classes, ev.target.value])
              ev.target.value = ''
              
              document.getElementById("class-input")?.blur();
              setUnsaved(true);
              return;
            }
          }}
        />
        <List>
          {classes.map((lab) => {
            if(lab === null) return;
            
            let style = {}
            if(lab == curr_lab) {
              const c               = _PALLETTE[classes.indexOf(lab)];
              style.color           = 'white';
              style.backgroundColor = `rgba(${c[0]},${c[1]},${c[2]},${c[3]})`
            }
            return <ListItem key={lab} style={style}> {lab} </ListItem>;
          })}
        </List>
      </div>

      {/* umap interface */}
      <div id="right_pane" className="split right">
        {unsaved ? <> <div style={{paddingLeft:"10px", color:"red"}}> UNSAVED - Press Cmd to save </div> </> : <></>}
        <div>
          <DeckGL
            views={[main_view]}
            layers={[umap_layer, ann_layer]}
            initialViewState={initialViewState}
            getCursor={() => "crosshair"}
          />
        </div>
      </div>
    </>
  );
}

export default withStreamlitConnection(UMap)


