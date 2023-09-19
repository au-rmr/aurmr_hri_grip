import { Button, Stack, Card, CardContent, Modal, Box, Typography, TextField, Slider } from "@mui/material";
import AddIcon from '@mui/icons-material/Add';
import React, { useState } from 'react';
import InputLabel from '@mui/material/InputLabel';
import MenuItem from '@mui/material/MenuItem';
import FormControl from '@mui/material/FormControl';
import Select, { SelectChangeEvent } from '@mui/material/Select';
import SortableList, { SortableItem } from "react-easy-sort";
import {arrayMoveImmutable} from "array-move";
import loadingSvg from './loading.svg';



const idb = require('./inventory.json');
// import React from 'react';
console.log(idb);

function shuffle(array: any[]): any[] {
    let currentIndex = array.length,  randomIndex;

    // While there remain elements to shuffle.
    while (currentIndex != 0) {

        // Pick a remaining element.
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;

        // And swap it with the current element.
        [array[currentIndex], array[randomIndex]] = [
        array[randomIndex], array[currentIndex]];
    }

    return array;
}

export function randomizeInventory(itemsPerBin: number = 4): any {
    let items = shuffle([...idb.items]);
    items = items.filter((item: any) => !item.hasOwnProperty('disabled') || item.disabled === false);
    let inventory: {[key: string]: any[]} = {
        '1A': [],
        '1B': [],
        '1C': []
    }
    
    // Start with 1A because large items can be there.
    for (let i = 0; i < itemsPerBin; i++) {
        let item = items.pop();
        if (item) {
            inventory['1A'].push(item);
        }
    }

    // Remove large items now
    items = items.filter((item: any) => !item.hasOwnProperty('large') || item.large === false);

    // Then fill 1B and 1C with the rest.
    for (let bin of ['1B', '1C']) {
        for (let i = 0; i < itemsPerBin; i++) {
            let item = items.pop();
            if (item) {
                inventory[bin].push(item);
            }
        }
    }

    return inventory;
};

const modalStyle = {
    position: 'absolute' as 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: 240,
    bgcolor: 'background.paper',
    border: '1px solid #444',
    boxShadow: 24,
    p: 4,
  };

const wideModalStyle = {
    position: 'absolute' as 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    width: 480,
    bgcolor: 'background.paper',
    border: '1px solid #444',
    boxShadow: 24,
    p: 4,
  };

const AddItemModal = (props: {inventory: any, bin: string, open: boolean, onClose: () => any, onSubmit: (itemId: number) => void}): any => {
    const [itemId, setItemId] = useState(4);
    const {open, onClose, onSubmit, inventory} = props;

    return (
    <Modal
        open={open}
        onClose={onClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
    >
        <Card sx={modalStyle}>
        <Typography id="modal-modal-title" variant="h6" component="h2">
           Add Item
        </Typography>
            <div>
            <Typography sx={{ mt: 2 }}>
            <FormControl fullWidth>
                <InputLabel>Item</InputLabel>
                <Select
                    labelId="demo-simple-select-label"
                    id="demo-simple-select"
                    value={itemId}
                    label="Age"
                    onChange={e => setItemId(parseInt(e.target.value as string))} 
                >
                    {
                        idb.items.filter((item: any) => !inventory['1A'].includes(item) && !inventory['1B'].includes(item) && !inventory['1C'].includes(item)).map((item: any) => {
                            return <MenuItem value={item.id}>{item.description}</MenuItem>
                        })
                    }
                </Select>
                </FormControl>
            </Typography>
            <Typography sx={{ mt: 2 }}>
                <Button color="primary" variant="contained" onClick={() => onSubmit(itemId)} style={{marginRight:'10px'}}>
                Add
                </Button>
                <Button color="error" variant="contained" onClick={() => onClose()}>
                Cancel
                </Button>
            </Typography>
            </div>
        </Card>
    </Modal>
    );
}

export const RandomizeInventoryModal = (props: {open: boolean, onClose: () => any, onRandomize: (maxItemsPerBin: number) => any}): any => {
    const [maxItemsPerBin, setMaxItemsPerBin] = useState(4);
    const {open, onClose, onRandomize} = props;

    return (
    <Modal
        open={open}
        onClose={onClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
    >
        <Card sx={modalStyle}>
        <Typography id="modal-modal-title" variant="h6" component="h2">
           Randomize Inventory
        </Typography>
            <div>
            <Typography sx={{ mt: 2 }}>
                <TextField value={maxItemsPerBin} onChange={e => setMaxItemsPerBin(parseInt(e.target.value))} id="outlined-basic" label="Max Items Per Bin" variant="outlined" type="number" />
            </Typography>
            <Typography sx={{ mt: 2 }}>
                <Button color="primary" variant="contained" onClick={() => onRandomize(maxItemsPerBin)} style={{marginRight:'10px'}}>
                Randomize
                </Button>
                <Button color="error" variant="contained" onClick={() => onClose()}>
                Cancel
                </Button>
            </Typography>
            </div>
        </Card>
    </Modal>
    );
}

export const ReviewPickOrderModal = (props: {open: boolean, onClose: () => any, onSubmit: (order: any, name: string) => any, inventory: any, loading: boolean}): any => {
    
    const {open, onClose, onSubmit, inventory, loading} = props;
    
    const [items, setItems] = React.useState<any[]>([]);
    const [name, setName] = React.useState<string>("");

    React.useEffect(() => {
        let itemList = [];
        for (let bin of ['1A', '1B', '1C']) {
            for (let item of props.inventory[bin]) {
                item.bin = bin;
                itemList.push(item);
            }
        }
        itemList = shuffle(itemList);
        setItems(itemList);
    }, [props.inventory]);

    const onSortEnd = (oldIndex: number, newIndex: number) => {
        setItems((array) => {
            console.log('arrayMoveImm');
            return arrayMoveImmutable(array, oldIndex, newIndex)
        });
    };

    return (
    <Modal
        open={open}
        onClose={onClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
    >
        <Card sx={wideModalStyle}>
        <Typography id="modal-modal-title" variant="h6" component="h2">
           Review Pick Order
        </Typography>
        {loading ? (
                      <Typography sx={{ mt: 2 }}>
                        <img src={loadingSvg} />
                      </Typography>
                    ) : (
            <div className='review-picks'>
                <div className='review-picks-items'>
                <SortableList
                    onSortEnd={onSortEnd}
                    className="review-picks-list"
                    draggedItemClassName="review-picks-dragged"
                    >
                    {items.map(item => (
                        <SortableItem key={item.id}>
                            <div className='review-picks-item'>
                                
                                    <div className='review-picks-bin'>{item.bin}</div>
                                    <img className='review-picks-img' src={"/item_images/" + String(item.id).padStart(3, '0') + '_front.jpg'} />
                                    <div className='review-picks-description'>{item.description}</div>
                                
                            </div>
                        </SortableItem>
                    ))}
                    </SortableList>
                
                </div>
            <Typography sx={{ mt: 2 }}>
                <TextField value={name} onChange={e => setName(e.target.value)} id="outlined-basic" label="Session Tag" variant="outlined" />
            </Typography>
            <Typography sx={{ mt: 2 }}>
                <Button color="success" variant="contained" onClick={() => onSubmit(items, name)} style={{marginRight:'10px'}}>
                Start
                </Button>
            </Typography>
            </div>
        )}
        </Card>
    </Modal>
    );
}

export const StreamlinedPickModal = (props: {
    open: boolean,
    onClose: () => any,
    onSubmit: (selectedItem: any, sessionName: string) => any,
    onFinished: (sessionName: string) => any,
    loading: boolean,
    sessionName: string | null
}): any => {

    const {open, onClose, onSubmit, onFinished, loading, sessionName} = props;
    const [name, setName] = React.useState<string>(sessionName || "");
    const [selectedItem, setSelectedItem] = React.useState<number | null>(null);
    const [selectedBin, setSelectedBin] = React.useState<string>("1C");

    let items = [...idb.items];
    items = items.filter((item: any) => !item.hasOwnProperty('disabled') || item.disabled === false);

    const handleSelectClicked = (selectedItem: number, sessionName: string) => {
        const selectedItemObj = items.filter((item) => item.id === selectedItem)[0];
        selectedItemObj.bin = selectedBin;
        onSubmit(selectedItemObj, sessionName);
    }

    return (
    <Modal
        open={open}
        onClose={onClose}
        aria-labelledby="modal-modal-title"
        aria-describedby="modal-modal-description"
    >
        <Card sx={wideModalStyle}>
        <Typography id="modal-modal-title" variant="h6" component="h2">
           Choose Target Item
        </Typography>
        {loading ? (
                      <Typography sx={{ mt: 2 }}>
                        <img src={loadingSvg} />
                      </Typography>
                    ) : (
            <div className='review-picks'>
                <div className='review-picks-items'>
                    {items.map((item: any) => (
                        <div onClick={() => setSelectedItem(item.id)} className={`review-picks-item ${item.id === selectedItem ? 'review-picks-item-selected' : ''} ${item.holdout ? 'review-picks-item-holdout' : ''}`}>
                                <div className='review-picks-bin'>{item.bin}</div>
                                <img className='review-picks-img' src={"/item_images/" + String(item.id).padStart(3, '0') + '_front.jpg'} />
                                <div className='review-picks-description'>{item.description}</div>
                        </div>
                    ))}
                
                </div>
            <Typography sx={{ mt: 2 }}>
                <TextField value={selectedBin} onChange={e => setSelectedBin(e.target.value)} id="outlined-basic" label="Bin" variant="outlined" />
            </Typography>
            <Typography sx={{ mt: 2 }}>
                <TextField value={name} onChange={e => setName(e.target.value)} id="outlined-basic" label="Session Tag" variant="outlined" />
            </Typography>
            <Typography sx={{ mt: 2 }}>
                <Button disabled={!selectedItem} color="success" variant="contained" onClick={() => handleSelectClicked(selectedItem!, name)} style={{marginRight:'10px'}}>
                Start
                </Button>

                <Button color="info" variant="contained" onClick={() => onFinished(name)} style={{marginRight:'10px'}}>
                Finish
                </Button>
            </Typography>
            </div>
        )}
        </Card>
    </Modal>
    );
}


export const InventoryManager = (props: {inventory: any, onUpdate: (inventory: any) => any}) => {

    let [addItemModalOpen, setAddItemModalOpen] = useState<{[key: string]: boolean}>({'1C': false, '1B': false, '1A': false});

    function clearItems(bin: string) {
        let inventory = {...props.inventory};
        inventory[bin] = [];
        props.onUpdate(inventory);
    }

    function addItem(bin: string, itemId: number) {
        let matchingItems = idb.items.filter((item: any) => item.id === itemId);
        if (matchingItems.length === 0) {
            return;
        }
        let inventory = {...props.inventory};
        inventory[bin].push(matchingItems[0]);
        props.onUpdate(inventory);

        setAddItemModalOpen({...addItemModalOpen, [bin]: false});

    }

    

    return (
    <div className='connected-inventory'>
    
                  <Typography variant="overline">
                      Manage Inventory:
                    </Typography>
                {['1C', '1B', '1A'].map((bin: string) => (
                    <div className='connected-inventory-bin'>
                    <Typography variant="h6">
                      {bin}
                    </Typography>
                    <Card>
                      <CardContent>
                        {(props.inventory.hasOwnProperty(bin) && props.inventory[bin].length > 0) ? (
                            props.inventory[bin].map((item: any) => (
                                <div className='inventory-item'>
                                    <div className='inventory-item-image'>
                                        <img style={{width:'60px', height:'60px'}} src={"/item_images/" + String(item.id).padStart(3, '0') + '_front.jpg'} />
                                    </div>
                                    <div className='inventory-item-description'>
                                        {item.description}
                                    </div>
                                </div>
                            ))
                        ) : null}
                        <Button size="small" color="primary" onClick={() => setAddItemModalOpen({...addItemModalOpen, [bin]: true})}>Add Item</Button>
                        <Button onClick={() => clearItems(bin)} size="small" color="error">Clear Items</Button>
                      </CardContent>
                    </Card>
                    <AddItemModal inventory={props.inventory} bin={bin} open={addItemModalOpen[bin]} onClose={() => setAddItemModalOpen({...addItemModalOpen, [bin]: false})} onSubmit={(itemId: number) => addItem(bin, itemId)} />
                  </div>
                ))}
                  

                  
    {idb.items.forEach((item: any) => (
        <div>{item.description}</div>
    ))}
  </div>
    );
};